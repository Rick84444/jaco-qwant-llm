"""
acfq_rig.py - activation sparsity on quantized weights, measured against
the best dense quantized kernel (the real opponent on a phone), with an optional energy measurement.

For every format (fp32, Q8, Q4) x every sparsity level (dense, s30..s70):
  err      relative L2 error vs float64 product with the ORIGINAL fp32 weights
           (so quantization error and sparsity error are both included)
  MB/tok   weight bytes actually read per token (rows read x row bytes)
  us       median per-token time over interleaved rounds (mask cost included)
  vs_opp   speedup relative to the opponent (see below)
  mJ/tok   (with --energy) net battery energy per token above idle

Formats q8h/q4h use the same bytes as q8/q4 with fp16 arithmetic (ARMv8.2).
Opponent = the fastest DENSE quantized format. A sparse config wins if it is
faster than the opponent and its error is within --err-budget of the
opponent's error.

  python3 acfq_rig.py --synthetic
  python3 acfq_rig.py --data acts            # from capture_activations.py
  python3 acfq_rig.py --synthetic --energy   # phone on battery, screen steady
"""
import argparse
import ctypes
import glob
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

FMT = {"f32": 0, "q8": 1, "q4": 2, "q8h": 3, "q4h": 4}
F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)
U8P = ctypes.POINTER(ctypes.c_uint8)


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.acfq_row_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.acfq_row_bytes.restype = ctypes.c_size_t
    lib.acfq_run.argtypes = [ctypes.c_int, U8P, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int,
                             ctypes.c_float, I32P, F32P]
    lib.acfq_run.restype = ctypes.c_long
    lib.acfq_bench.argtypes = [ctypes.c_int, U8P, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int,
                               ctypes.c_float, ctypes.c_int, I32P, F32P]
    lib.acfq_bench.restype = ctypes.c_double
    lib.acfq_run_for.argtypes = [ctypes.c_int, U8P, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int,
                                 ctypes.c_float, ctypes.c_double, I32P, F32P,
                                 ctypes.POINTER(ctypes.c_double)]
    lib.acfq_run_for.restype = ctypes.c_long
    lib.acfq_has_fp16.restype = ctypes.c_int
    return lib


def fp(a):
    return a.ctypes.data_as(F32P)


def ip(a):
    return a.ctypes.data_as(I32P)


def up(a):
    return a.ctypes.data_as(U8P)


# ---------------------------------------------------------------- quantization

def pack_f32(W):
    return np.ascontiguousarray(W.astype(np.float32)).view(np.uint8).reshape(W.shape[0], -1)


def quant_q8(W):
    K, N = W.shape
    g = W.reshape(K, N // 32, 32).astype(np.float32)
    amax = np.abs(g).max(axis=2)
    s = (amax / 127.0).astype(np.float16)
    sf = s.astype(np.float32)
    q = np.where(sf[..., None] > 0, np.rint(g / np.where(sf > 0, sf, 1)[..., None]), 0)
    q = np.clip(q, -127, 127).astype(np.int8).reshape(K, N)
    return np.concatenate([s.view(np.uint8).reshape(K, -1), q.view(np.uint8)], axis=1)


def quant_q4(W):
    """Q4_0-style: d = (value with max |w|) / -8, q = clamp(round(w/d) + 8, 0, 15)."""
    K, N = W.shape
    g = W.reshape(K, N // 32, 32).astype(np.float32)
    pick = np.take_along_axis(g, np.abs(g).argmax(axis=2)[..., None], axis=2)[..., 0]
    s = (pick / -8.0).astype(np.float16)
    sf = s.astype(np.float32)
    q = np.where(sf[..., None] != 0, np.rint(g / np.where(sf != 0, sf, 1)[..., None]), 0) + 8
    q = np.clip(q, 0, 15).astype(np.uint8)
    packed = (q[..., :16] | (q[..., 16:] << 4)).reshape(K, N // 2)
    return np.concatenate([s.view(np.uint8).reshape(K, -1), packed], axis=1)


QUANT = {"f32": pack_f32, "q8": quant_q8, "q4": quant_q4, "q8h": quant_q8, "q4h": quant_q4}


# ---------------------------------------------------------------- data

def synthetic_layers(K, N, T, seed):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)
    X = (rng.standard_t(3, (T, K)) * 0.3).astype(np.float32)
    X[:, rng.choice(K, 8, replace=False)] *= 20.0
    return [("synthetic_%dx%d" % (K, N), W, X)]


def dir_layers(path, max_tokens):
    out = []
    for wf in sorted(glob.glob(os.path.join(path, "W_*.npy"))):
        name = os.path.basename(wf)[2:-4]
        xf = os.path.join(path, "X_%s.npy" % name)
        if not os.path.exists(xf):
            continue
        W = np.load(wf).astype(np.float32)
        X = np.ascontiguousarray(np.load(xf).astype(np.float32)[:max_tokens])
        if W.shape[0] != X.shape[1] or W.shape[1] % 32:
            print("skip %s: W %s X %s (N must be a multiple of 32)" % (name, W.shape, X.shape),
                  file=sys.stderr)
            continue
        out.append((name, W, X))
    return out


# ---------------------------------------------------------------- energy

def power_reader(sysfs_dir):
    """Returns (name, fn) where fn() -> watts, or (None, None)."""
    cur = os.path.join(sysfs_dir, "current_now")
    vol = os.path.join(sysfs_dir, "voltage_now")

    def read_sysfs():
        with open(cur) as f:
            c = abs(int(f.read().strip()))
        with open(vol) as f:
            v = int(f.read().strip())
        return c * 1e-6 * v * 1e-6

    try:
        read_sysfs()
        return "sysfs:%s" % sysfs_dir, read_sysfs
    except (OSError, ValueError):
        pass

    def read_api():
        r = subprocess.run(["termux-battery-status"], capture_output=True, text=True, timeout=5)
        d = json.loads(r.stdout)
        return abs(float(d["current"])) * 1e-6 * 3.85

    try:
        read_api()
        return "termux-battery-status (current only, nominal 3.85 V, slow updates)", read_api
    except Exception:
        return None, None


def battery_status(sysfs_dir):
    try:
        with open(os.path.join(sysfs_dir, "status")) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def sample_during(read, fn, hz):
    samples, stop = [], threading.Event()

    def loop():
        while not stop.is_set():
            try:
                samples.append(read())
            except Exception:
                pass
            stop.wait(1.0 / hz)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    result = fn()
    stop.set()
    th.join()
    return result, samples


# ---------------------------------------------------------------- measurement

def run_accuracy(lib, fmt, Wq, K, N, X, tau, Yref):
    T = X.shape[0]
    idx = np.empty(K, dtype=np.int32)
    Y = np.empty((T, N), dtype=np.float32)
    active = lib.acfq_run(FMT[fmt], up(Wq), K, N, fp(X), T, ctypes.c_float(tau), ip(idx), fp(Y))
    diff = np.linalg.norm(Y - Yref, axis=1)
    ref = np.linalg.norm(Yref, axis=1)
    rel = diff / np.where(ref > 0, ref, 1)
    return {
        "err": float(rel.mean()),
        "err_p95": float(np.quantile(rel, 0.95)),
        "sparsity": float(1.0 - active / (T * K)),
        "active_rows": active / T,
    }


def run_layer(lib, name, W, X, args):
    K, N = W.shape
    T = X.shape[0]
    n_cal = max(1, int(T * args.calib_frac))
    Xc, Xe = X[:n_cal], np.ascontiguousarray(X[n_cal:] if T > n_cal else X)
    Yref = (Xe.astype(np.float64) @ W.astype(np.float64)).astype(np.float64)

    packed = {f: np.ascontiguousarray(QUANT[f](W)) for f in args.formats}
    taus = [("dense", -1.0)] + [("s%02d" % round(s * 100), float(np.quantile(np.abs(Xc), s)))
                                for s in args.sparsities]
    configs = [(f, label, tau) for f in args.formats for label, tau in taus]

    rows = {}
    for f, label, tau in configs:
        r = run_accuracy(lib, f, packed[f], K, N, Xe, tau, Yref)
        rb = lib.acfq_row_bytes(FMT[f], N)
        r.update({"fmt": f, "mask": label, "tau": tau, "row_bytes": rb,
                  "mb_per_token": r["active_rows"] * rb / 1e6})
        rows[(f, label)] = r

    Tb = min(Xe.shape[0], args.bench_tokens)
    Xb = np.ascontiguousarray(Xe[:Tb])
    idx = np.empty(K, dtype=np.int32)
    y = np.empty(N, dtype=np.float32)
    times = {c[:2]: [] for c in configs}
    order_rng = np.random.default_rng(args.seed)
    for f, label, tau in configs:
        lib.acfq_bench(FMT[f], up(packed[f]), K, N, fp(Xb), Tb, ctypes.c_float(tau), 1, ip(idx), fp(y))
    for _ in range(args.rounds):
        for ci in order_rng.permutation(len(configs)):
            f, label, tau = configs[ci]
            ns = lib.acfq_bench(FMT[f], up(packed[f]), K, N, fp(Xb), Tb, ctypes.c_float(tau),
                                args.reps, ip(idx), fp(y))
            times[(f, label)].append(ns / 1e3)
    for key, ts in times.items():
        rows[key]["us"] = statistics.median(ts)
        rows[key]["us_spread"] = (max(ts) - min(ts)) / statistics.median(ts)

    if args.energy:
        name_, read = power_reader(args.power_dir)
        if read is None:
            print("   energy: no readable power source (sysfs or termux-battery-status)")
        else:
            status = battery_status(args.power_dir)
            print("   energy source: %s  battery status: %s" % (name_, status))
            if status.lower() not in ("discharging", "not charging", "unknown"):
                print("   WARNING: battery is %s; charger current makes energy numbers invalid" % status)
            _, idle = sample_during(read, lambda: time.sleep(args.energy_seconds), args.energy_hz)
            p_idle = statistics.mean(idle) if idle else float("nan")
            for f, label, tau in configs:
                el = ctypes.c_double(0.0)
                done, samp = sample_during(
                    read,
                    lambda: lib.acfq_run_for(FMT[f], up(packed[f]), K, N, fp(Xb), Tb,
                                             ctypes.c_float(tau), args.energy_seconds,
                                             ip(idx), fp(y), ctypes.byref(el)),
                    args.energy_hz)
                p = statistics.mean(samp) if samp else float("nan")
                rows[(f, label)].update({
                    "watts": p, "watts_idle": p_idle, "energy_samples": len(samp),
                    "mj_per_token": (p - p_idle) * el.value / done * 1e3 if done else float("nan"),
                })

    dense_q = [r for r in rows.values() if r["mask"] == "dense" and r["fmt"] != "f32"]
    base = min(dense_q, key=lambda r: r["us"]) if dense_q else rows.get(("f32", "dense"))
    for r in rows.values():
        r["vs_opp"] = base["us"] / r["us"]
        r["err_vs_opp"] = r["err"] - base["err"]
        r["win"] = bool(r["fmt"] != "f32" and r["mask"] != "dense" and r["vs_opp"] > 1.0
                        and r["err_vs_opp"] <= args.err_budget)
    winners = [r for r in rows.values() if r.get("win")]
    best = max(winners, key=lambda r: r["vs_opp"]) if winners else None
    return {"layer": name, "K": K, "N": N, "tokens_eval": int(Xe.shape[0]),
            "tokens_calib": int(Xc.shape[0]), "bench_tokens": Tb,
            "opponent": "%s dense" % base["fmt"], "results": list(rows.values()), "best": best}


def print_layer(rep, energy):
    print("\n== %s  K=%d N=%d  eval=%d calib=%d  opponent: %s" % (
        rep["layer"], rep["K"], rep["N"], rep["tokens_eval"], rep["tokens_calib"], rep["opponent"]))
    hdr = "   %-4s %-6s %8s %7s %7s %8s %8s %7s" % (
        "fmt", "mask", "sparsity", "err", "d_err", "MB/tok", "us", "vs_opp")
    if energy:
        hdr += " %8s" % "mJ/tok"
    print(hdr)
    for r in rep["results"]:
        line = "   %-4s %-6s %8.3f %7.3f %+7.3f %8.3f %8.1f %6.2fx" % (
            r["fmt"], r["mask"], r["sparsity"], r["err"], r["err_vs_opp"],
            r["mb_per_token"], r["us"], r["vs_opp"])
        if energy:
            line += " %8.4f" % r.get("mj_per_token", float("nan"))
        if r.get("win"):
            line += "  win"
        if r["us_spread"] > 0.15:
            line += "  (noisy %.0f%%)" % (r["us_spread"] * 100)
        print(line)
    b = rep["best"]
    if b:
        print("   BEST vs %s: %s %s  %.2fx faster, err %+.3f, %.2f MB/tok" % (
            rep["opponent"], b["fmt"], b["mask"], b["vs_opp"], b["err_vs_opp"], b["mb_per_token"]))
    else:
        print("   no sparse config beats %s within the error budget" % rep["opponent"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--data")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--lib", default=os.path.join(here, "libacfq.so"))
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--formats", nargs="+", default=["f32", "q8", "q4", "q8h", "q4h"], choices=list(FMT))
    ap.add_argument("--sparsities", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--calib-frac", type=float, default=0.25)
    ap.add_argument("--bench-tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--err-budget", type=float, default=0.03,
                    help="allowed error increase over fastest dense quantized format (relative-L2 units)")
    ap.add_argument("--energy", action="store_true")
    ap.add_argument("--energy-seconds", type=float, default=4.0)
    ap.add_argument("--energy-hz", type=float, default=20.0)
    ap.add_argument("--power-dir", default="/sys/class/power_supply/battery")
    ap.add_argument("--json", default="acfq_report.json")
    args = ap.parse_args()

    lib = load_lib(args.lib)
    if args.synthetic:
        layers = synthetic_layers(args.K, args.N, args.tokens, args.seed)
        source = "SYNTHETIC (random weights, heavy-tailed activations)"
    else:
        layers = dir_layers(args.data, args.max_tokens)
        source = "USER_SUPPLIED_NPY (provenance unverified) from %s" % args.data
        if not layers:
            sys.exit("no usable W_*.npy / X_*.npy pairs in %s" % args.data)
    print("source: %s" % source)
    print("fp16 arithmetic kernels: %s" % ("on" if lib.acfq_has_fp16() else "off (q8h/q4h run as q8/q4)"))
    print("opponent = fastest dense quantized format; win = faster with error increase <= %.3f" % args.err_budget)
    reports = []
    for name, W, X in layers:
        rep = run_layer(lib, name, W, X, args)
        print_layer(rep, args.energy)
        reports.append(rep)
    with open(args.json, "w") as f:
        json.dump({"source": source, "args": vars(args), "layers": reports}, f, indent=2, default=float)
    print("\nreport: %s" % args.json)


if __name__ == "__main__":
    main()
