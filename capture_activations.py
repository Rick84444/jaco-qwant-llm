"""
capture_activations.py - real activations for acf_rig.py, plus an end-to-end
perplexity test with activation sparsity applied inside the model.

Runs on a PC with torch + transformers (GPU optional).

  pip install torch transformers
  # capture inputs and weights of selected linear layers
  python capture_activations.py capture --model Qwen/Qwen2.5-0.5B --text corpus.txt --out acts
  # end-to-end: perplexity dense vs per-channel threshold sparsity on every linear input
  python capture_activations.py eval --model Qwen/Qwen2.5-0.5B --text corpus.txt --sparsities 0.3 0.4 0.5
  # same, with weights in the Q4 layout the phone kernel uses
  python capture_activations.py eval --model Qwen/Qwen2.5-0.5B --wquant q4

Capture writes, per layer:
  W_<name>.npy   (in_features x out_features) float32, matches acf_rig layout
  X_<name>.npy   (tokens x in_features) float32, the actual inputs to that layer

Eval calibrates one threshold per linear layer on the first --calib-frac of the
token stream (quantile of |x| = target sparsity), then measures perplexity on
the remainder with inputs below the threshold set to zero. The zeroing is done
in float, so this measures quality only; speed is measured by acf_rig.py.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return tok, model.to(device).eval()


def read_text(path):
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(ds["text"])
    except Exception as exc:
        sys.exit("no --text given and wikitext unavailable (%s)" % exc)


def token_ids(tok, text, max_tokens):
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids[:max_tokens]


def linear_layers(model, layer_filter, proj_filter):
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        parts = name.split(".")
        layer_no = next((int(p) for p in parts if p.isdigit()), None)
        if layer_filter is not None and layer_no not in layer_filter:
            continue
        if proj_filter and parts[-1] not in proj_filter:
            continue
        out.append((name, mod))
    return out


def run_chunks(model, ids, device, ctx):
    """Feed ids in non-overlapping windows of ctx tokens; returns summed NLL and count."""
    nll, count = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(ids) - 1, ctx):
            chunk = ids[s:s + ctx + 1].unsqueeze(0).to(device)
            if chunk.shape[1] < 2:
                break
            logits = model(chunk[:, :-1]).logits.float()
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1), reduction="sum")
            nll += float(loss)
            count += chunk.shape[1] - 1
    return nll, count


# ---------------------------------------------------------------- capture

def cmd_capture(args):
    device = pick_device(args.device)
    tok, model = load_model(args.model, device)
    ids = token_ids(tok, read_text(args.text), args.max_tokens)
    layers = linear_layers(model, set(args.layers) if args.layers else None, set(args.projs))
    if not layers:
        sys.exit("no linear layers matched")
    store = {name: [] for name, _ in layers}

    def make_hook(name):
        def hook(_mod, inputs):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            store[name].append(x.cpu().numpy())
        return hook

    handles = [mod.register_forward_pre_hook(make_hook(name)) for name, mod in layers]
    run_chunks(model, ids, device, args.ctx)
    for h in handles:
        h.remove()

    os.makedirs(args.out, exist_ok=True)
    for name, mod in layers:
        key = name.replace(".", "_")
        X = np.concatenate(store[name], axis=0)
        W = mod.weight.detach().float().cpu().numpy().T.copy()
        np.save(os.path.join(args.out, "X_%s.npy" % key), X.astype(np.float32))
        np.save(os.path.join(args.out, "W_%s.npy" % key), W.astype(np.float32))
        print("%-45s W %s  X %s" % (key, W.shape, X.shape))
    print("saved %d layers to %s" % (len(layers), args.out))


# ---------------------------------------------------------------- eval

def fake_quant_(layers, kind):
    """Replace each weight with its dequantized Q8/Q4 version, using the same
    layout as acfq_kernel.c: W stored as (in x out), groups of 32 along out,
    one fp16 scale per group. torch weight is (out x in), so groups run along
    dim 0 for each input column."""
    with torch.no_grad():
        for _name, mod in layers:
            w = mod.weight.data.float()
            out_f, in_f = w.shape
            if out_f % 32:
                continue
            g = w.t().reshape(in_f, out_f // 32, 32)
            if kind == "q8":
                s = (g.abs().amax(dim=2, keepdim=True) / 127.0).half().float()
                q = torch.where(s > 0, torch.round(g / torch.where(s > 0, s, torch.ones_like(s))),
                                torch.zeros_like(g)).clamp(-127, 127)
                deq = q * s
            else:
                pick = torch.gather(g, 2, g.abs().argmax(dim=2, keepdim=True))
                s = (pick / -8.0).half().float()
                q = torch.where(s != 0, torch.round(g / torch.where(s != 0, s, torch.ones_like(s))),
                                torch.zeros_like(g)) + 8
                deq = (q.clamp(0, 15) - 8) * s
            mod.weight.data.copy_(deq.reshape(in_f, out_f).t().to(mod.weight.dtype))


def calibrate(model, layers, ids, device, ctx, sparsity, max_samples):
    """Per-layer tau = quantile(|x|, sparsity) over calibration tokens."""
    samples = {name: [] for name, _ in layers}

    def make_hook(name):
        def hook(_mod, inputs):
            x = inputs[0].detach().float().abs().reshape(-1)
            if x.numel() > max_samples:
                x = x[torch.randint(0, x.numel(), (max_samples,), device=x.device)]
            samples[name].append(x.cpu())
        return hook

    handles = [mod.register_forward_pre_hook(make_hook(name)) for name, mod in layers]
    run_chunks(model, ids, device, ctx)
    for h in handles:
        h.remove()
    return {name: float(torch.quantile(torch.cat(v)[:max_samples * 8], sparsity))
            for name, v in samples.items()}


def eval_sparse(model, layers, taus, ids, device, ctx):
    kept = {"n": 0, "total": 0}

    def make_hook(tau):
        def hook(_mod, inputs):
            x = inputs[0]
            m = x.abs() > tau
            kept["n"] += int(m.sum())
            kept["total"] += m.numel()
            return (x * m,) + tuple(inputs[1:])
        return hook

    handles = [mod.register_forward_pre_hook(make_hook(taus[name])) for name, mod in layers]
    nll, count = run_chunks(model, ids, device, ctx)
    for h in handles:
        h.remove()
    return math.exp(nll / count), 1.0 - kept["n"] / max(1, kept["total"])


def cmd_eval(args):
    device = pick_device(args.device)
    tok, model = load_model(args.model, device)
    ids = token_ids(tok, read_text(args.text), args.max_tokens)
    n_cal = int(len(ids) * args.calib_frac)
    ids_cal, ids_eval = ids[:n_cal], ids[n_cal:]
    layers = linear_layers(model, set(args.layers) if args.layers else None, set(args.projs))
    print("model %s  device %s  layers %d  calib %d  eval %d tokens" % (
        args.model, device, len(layers), len(ids_cal), len(ids_eval)))

    nll, count = run_chunks(model, ids_eval, device, args.ctx)
    ppl_fp = math.exp(nll / count)
    print("%-10s ppl %8.3f" % ("dense fp", ppl_fp))
    if args.wquant != "none":
        fake_quant_(layers, args.wquant)
        nll, count = run_chunks(model, ids_eval, device, args.ctx)
        ppl_q = math.exp(nll / count)
        print("%-10s ppl %8.3f  (%+.1f%% vs fp)  <- reference for the rows below" % (
            "dense " + args.wquant, ppl_q, (ppl_q / ppl_fp - 1) * 100))
    ppl_dense = math.exp(nll / count)
    results = [{"sparsity_target": 0.0, "sparsity_actual": 0.0, "ppl": ppl_dense, "ppl_rel": 0.0}]
    for s in args.sparsities:
        taus = calibrate(model, layers, ids_cal, device, args.ctx, s, args.calib_samples)
        ppl, actual = eval_sparse(model, layers, taus, ids_eval, device, args.ctx)
        rel = ppl / ppl_dense - 1.0
        print("s=%.2f     ppl %8.3f  (%+.1f%%)  actual sparsity %.3f" % (s, ppl, rel * 100, actual))
        results.append({"sparsity_target": s, "sparsity_actual": actual, "ppl": ppl, "ppl_rel": rel})
    with open(args.json, "w") as f:
        json.dump({"model": args.model, "projs": args.projs, "wquant": args.wquant,
                   "ppl_fp": ppl_fp, "results": results}, f, indent=2)
    print("report: %s" % args.json)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="HF id or local path")
        p.add_argument("--text", help="UTF-8 text file; default wikitext-2 test (needs `datasets`)")
        p.add_argument("--device", default="auto")
        p.add_argument("--ctx", type=int, default=512)
        p.add_argument("--layers", type=int, nargs="*", help="layer numbers; default all")
        p.add_argument("--projs", nargs="+",
                       default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cap = sub.choices["capture"]
    cap.add_argument("--max-tokens", type=int, default=512)
    cap.add_argument("--out", default="acts")
    cap.set_defaults(layers=None)
    ev = sub.choices["eval"]
    ev.add_argument("--max-tokens", type=int, default=16384)
    ev.add_argument("--calib-frac", type=float, default=0.25)
    ev.add_argument("--calib-samples", type=int, default=200000)
    ev.add_argument("--sparsities", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    ev.add_argument("--wquant", choices=["none", "q8", "q4"], default="none",
                    help="fake-quantize weights in the acfq layout before measuring sparsity")
    ev.add_argument("--json", default="acf_eval.json")
    args = ap.parse_args()
    if args.cmd == "capture" and args.layers is None:
        args.layers = [0, 6, 12]
    {"capture": cmd_capture, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
