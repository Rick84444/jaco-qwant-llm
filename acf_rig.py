"""
acf_rig.py - accuracy and speed rig for activation-sparse GEMV.

Every sparse strategy is measured against the dense product on the same
weights and the same activations:
  rel_err = ||y_sparse - y_dense|| / ||y_dense||   (per token, then mean / p95)
  cos     = cosine(y_sparse, y_dense)
  speedup = dense_us / sparse_us  (mask cost included, timed inside C)

Data sources:
  --synthetic            random weights + heavy-tailed activations (no real model)
  --data DIR             W_<name>.npy (K x N) and X_<name>.npy (T x K) pairs,
                         produced by capture_activations.py from a real model

Strategies:
  energy_fixed     original block-energy mask, fixed target, fresh every token
  energy_dynamic   original entropy-scaled target, fresh every token
  energy_stale32   original backend as written: dynamic, recomputed every 32 tokens
  thr_sXX          per-channel |x| > tau, tau calibrated on held-out tokens to
                   reach XX% sparsity (TEAL-style)

Pass criterion (default): mean rel_err < 0.05 at sparsity >= 0.40.
"""
import argparse
import ctypes
import glob
import json
import os
import sys

import numpy as np

F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.acf_gemv_dense.argtypes = [F32P, ctypes.c_int, ctypes.c_int, F32P, F32P]
    lib.acf_gemv_rows.argtypes = [F32P, ctypes.c_int, ctypes.c_int, F32P, I32P, ctypes.c_int, F32P]
    lib.acf_mask_threshold.argtypes = [F32P, ctypes.c_int, ctypes.c_float, I32P]
    lib.acf_mask_threshold.restype = ctypes.c_int
    lib.acf_mask_energy_blocks.argtypes = [F32P, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                           ctypes.c_float, ctypes.c_int, F32P, I32P, I32P, F32P, F32P]
    lib.acf_mask_energy_blocks.restype = ctypes.c_int
    lib.acf_bench_dense.argtypes = [F32P, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int, ctypes.c_int, F32P]
    lib.acf_bench_dense.restype = ctypes.c_double
    lib.acf_bench_threshold.argtypes = [F32P, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int,
                                        ctypes.c_float, ctypes.c_int, F32P, I32P]
    lib.acf_bench_threshold.restype = ctypes.c_double
    lib.acf_bench_energy.argtypes = [F32P, ctypes.c_int, ctypes.c_int, F32P, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, F32P, I32P, F32P, I32P]
    lib.acf_bench_energy.restype = ctypes.c_double
    return lib


def fp(a):
    return a.ctypes.data_as(F32P)


def ip(a):
    return a.ctypes.data_as(I32P)


# ---------------------------------------------------------------- data

def synthetic_layers(K, N, T, seed):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((K, N)) * (1.0 / np.sqrt(K))).astype(np.float32)
    X = (rng.standard_t(3, (T, K)) * 0.3).astype(np.float32)
    outliers = rng.choice(K, 8, replace=False)
    X[:, outliers] *= 20.0
    return [("synthetic_%dx%d" % (K, N), W, X)]


def dir_layers(path, max_tokens):
    out = []
    for wf in sorted(glob.glob(os.path.join(path, "W_*.npy"))):
        name = os.path.basename(wf)[2:-4]
        xf = os.path.join(path, "X_%s.npy" % name)
        if not os.path.exists(xf):
            continue
        W = np.ascontiguousarray(np.load(wf).astype(np.float32))
        X = np.ascontiguousarray(np.load(xf).astype(np.float32)[:max_tokens])
        if W.shape[0] != X.shape[1]:
            print("skip %s: W %s vs X %s" % (name, W.shape, X.shape), file=sys.stderr)
            continue
        out.append((name, W, X))
    return out


# ---------------------------------------------------------------- masks

def masks_threshold(lib, X, tau):
    T, K = X.shape
    idx = np.empty(K, dtype=np.int32)
    out = []
    for t in range(T):
        x = np.ascontiguousarray(X[t])
        n = lib.acf_mask_threshold(fp(x), K, ctypes.c_float(tau), ip(idx))
        out.append(idx[:n].copy())
    return out


def masks_energy(lib, X, B, base, max_target, dynamic, interval):
    T, K = X.shape
    M = (K + B - 1) // B
    se = np.empty(M, dtype=np.float32)
    so = np.empty(M, dtype=np.int32)
    idx = np.empty(K, dtype=np.int32)
    out, cur = [], None
    for t in range(T):
        if t % interval == 0:
            x = np.ascontiguousarray(X[t])
            n = lib.acf_mask_energy_blocks(fp(x), K, B, base, max_target, dynamic,
                                           fp(se), ip(so), ip(idx), None, None)
            cur = idx[:n].copy()
        out.append(cur)
    return out


def calibrate_tau(X_calib, sparsity):
    return float(np.quantile(np.abs(X_calib), sparsity))


# ---------------------------------------------------------------- metrics

def errors(W, X, masks):
    Yd = X @ W
    rel, cos, act = [], [], []
    for t, idx in enumerate(masks):
        x = X[t]
        ys = x[idx] @ W[idx]
        yd = Yd[t]
        nd = np.linalg.norm(yd)
        rel.append(np.linalg.norm(ys - yd) / nd if nd > 0 else 0.0)
        ns = np.linalg.norm(ys)
        cos.append(float(ys @ yd / (ns * nd)) if ns > 0 and nd > 0 else 0.0)
        act.append(len(idx) / X.shape[1])
    rel = np.array(rel)
    return {
        "rel_err_mean": float(rel.mean()),
        "rel_err_p95": float(np.quantile(rel, 0.95)),
        "cos_mean": float(np.mean(cos)),
        "sparsity": float(1.0 - np.mean(act)),
    }


def check_kernel(lib, W, X):
    K, N = W.shape
    x = np.ascontiguousarray(X[0])
    y = np.empty(N, dtype=np.float32)
    lib.acf_gemv_dense(fp(W), K, N, fp(x), fp(y))
    ref = x.astype(np.float64) @ W.astype(np.float64)
    e_dense = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
    idx = np.arange(0, K, 3, dtype=np.int32)
    lib.acf_gemv_rows(fp(W), K, N, fp(x), ip(idx), len(idx), fp(y))
    ref = x[idx].astype(np.float64) @ W[idx].astype(np.float64)
    e_rows = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
    return e_dense, e_rows


# ---------------------------------------------------------------- run

def run_layer(lib, name, W, X, args):
    K, N = W.shape
    T = X.shape[0]
    n_cal = max(1, int(T * args.calib_frac))
    Xc, Xt = X[:n_cal], np.ascontiguousarray(X[n_cal:])
    if Xt.shape[0] == 0:
        Xc, Xt = X, X
    e_dense, e_rows = check_kernel(lib, W, Xt)
    Y = np.empty(N, dtype=np.float32)
    idx = np.empty(K, dtype=np.int32)
    M = (K + args.block - 1) // args.block
    se = np.empty(M, dtype=np.float32)
    so = np.empty(M, dtype=np.int32)
    Tb = min(Xt.shape[0], args.bench_tokens)
    Xb = np.ascontiguousarray(Xt[:Tb])

    lib.acf_bench_dense(fp(W), K, N, fp(Xb), Tb, 1, fp(Y))
    dense_ns = lib.acf_bench_dense(fp(W), K, N, fp(Xb), Tb, args.reps, fp(Y))

    rows = []

    def add(label, metrics, ns):
        metrics.update({"strategy": label, "us": ns / 1e3, "speedup": dense_ns / ns})
        rows.append(metrics)

    energy_cfgs = [
        ("energy_fixed", 0, 1),
        ("energy_dynamic", 1, 1),
        ("energy_stale32", 1, 32),
    ]
    for label, dyn, interval in energy_cfgs:
        m = masks_energy(lib, Xt, args.block, args.energy_base, 0.92, dyn, interval)
        ns = lib.acf_bench_energy(fp(W), K, N, fp(Xb), Tb, args.block, args.energy_base, 0.92,
                                  dyn, interval, args.reps, fp(Y), ip(idx), fp(se), ip(so))
        add(label, errors(W, Xt, m), ns)

    for s in args.sparsities:
        tau = calibrate_tau(Xc, s)
        m = masks_threshold(lib, Xt, tau)
        ns = lib.acf_bench_threshold(fp(W), K, N, fp(Xb), Tb, ctypes.c_float(tau),
                                     args.reps, fp(Y), ip(idx))
        r = errors(W, Xt, m)
        r["tau"] = tau
        add("thr_s%02d" % int(round(s * 100)), r, ns)

    passing = [r for r in rows
               if r["rel_err_mean"] < args.max_err and r["sparsity"] >= args.min_sparsity]
    best = max(passing, key=lambda r: r["speedup"]) if passing else None
    return {
        "layer": name, "K": K, "N": N, "tokens_eval": int(Xt.shape[0]),
        "tokens_calib": int(Xc.shape[0]), "bench_tokens": Tb,
        "kernel_check": {"dense_rel_err": e_dense, "rows_rel_err": e_rows},
        "dense_us": dense_ns / 1e3, "results": rows,
        "pass": best is not None, "best": best,
    }


def print_layer(rep, args):
    kc = rep["kernel_check"]
    print("\n== %s  K=%d N=%d  eval=%d calib=%d tokens" % (
        rep["layer"], rep["K"], rep["N"], rep["tokens_eval"], rep["tokens_calib"]))
    print("   kernel vs float64: dense %.1e, rows %.1e   dense %.1f us/token" % (
        kc["dense_rel_err"], kc["rows_rel_err"], rep["dense_us"]))
    print("   %-15s %8s %8s %8s %7s %8s %7s" % (
        "strategy", "sparsity", "err_mean", "err_p95", "cos", "us", "speedup"))
    for r in rep["results"]:
        flag = " ok" if (r["rel_err_mean"] < args.max_err and r["sparsity"] >= args.min_sparsity) else ""
        print("   %-15s %8.3f %8.3f %8.3f %7.4f %8.1f %6.2fx%s" % (
            r["strategy"], r["sparsity"], r["rel_err_mean"], r["rel_err_p95"],
            r["cos_mean"], r["us"], r["speedup"], flag))
    b = rep["best"]
    if b:
        print("   PASS  best: %s  sparsity %.2f  err %.3f  %.2fx" % (
            b["strategy"], b["sparsity"], b["rel_err_mean"], b["speedup"]))
    else:
        print("   FAIL  no strategy has err < %.2f at sparsity >= %.2f" % (args.max_err, args.min_sparsity))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--data", help="directory with W_*.npy / X_*.npy")
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "libacf_rig.so"))
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--calib-frac", type=float, default=0.25)
    ap.add_argument("--bench-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--energy-base", type=float, default=0.55)
    ap.add_argument("--sparsities", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--max-err", type=float, default=0.05)
    ap.add_argument("--min-sparsity", type=float, default=0.40)
    ap.add_argument("--json", default="acf_report.json")
    args = ap.parse_args()

    lib = load_lib(args.lib)
    if args.synthetic:
        layers = synthetic_layers(args.K, args.N, args.tokens, args.seed)
        source = "SYNTHETIC (random weights, heavy-tailed activations)"
    else:
        layers = dir_layers(args.data, args.max_tokens)
        source = "CAPTURED from %s" % args.data
        if not layers:
            sys.exit("no W_*.npy / X_*.npy pairs in %s" % args.data)

    print("source: %s" % source)
    print("criterion: rel_err_mean < %.2f at sparsity >= %.2f" % (args.max_err, args.min_sparsity))
    reports = []
    for name, W, X in layers:
        rep = run_layer(lib, name, W, X, args)
        print_layer(rep, args)
        reports.append(rep)

    n_pass = sum(r["pass"] for r in reports)
    print("\nlayers passing: %d / %d" % (n_pass, len(reports)))
    with open(args.json, "w") as f:
        json.dump({"source": source, "args": vars(args), "layers": reports}, f, indent=2)
    print("report: %s" % args.json)


if __name__ == "__main__":
    main()
