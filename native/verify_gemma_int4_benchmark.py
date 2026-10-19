#!/usr/bin/env python3
"""Validate a Gemma-shaped INT8xINT4 benchmark CSV (synthetic data).

Checks internal consistency only; it cannot prove the device provenance,
actual LiteRT performance or end-to-end Gemma token acceleration.
"""
import argparse
import csv
import io
import math
from pathlib import Path

CSV_COLUMNS = (
    "K", "N", "target_zero_pct", "actual_zero_pct", "scalar_us",
    "acfq_dense_us", "acfq_exact_zero_us", "dense_vs_scalar",
    "exact_zero_vs_scalar", "dense_abs_error", "exact_zero_abs_error",
    "dense_neon_blocks", "exact_zero_skipped_products",
)
INT_COLUMNS = (
    "K", "N", "target_zero_pct", "dense_neon_blocks",
    "exact_zero_skipped_products",
)
FLOAT_COLUMNS = tuple(key for key in CSV_COLUMNS if key not in INT_COLUMNS)
DEFAULT_CSV = (Path(__file__).resolve().parent.parent
               / "evidence/gemma-int4-pixel10pro-2026-10-19.csv")


class BenchmarkError(ValueError):
    pass


def check_report(text):
    lines = [line for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(io.StringIO("\n".join(lines) + "\n"))
    if reader.fieldnames != list(CSV_COLUMNS):
        raise BenchmarkError("wrong benchmark CSV schema")
    results = []
    seen = set()
    for raw in reader:
        if None in raw or any(raw.get(k) is None for k in CSV_COLUMNS):
            raise BenchmarkError("missing or extra CSV fields")
        try:
            row = {key: int(raw[key]) for key in INT_COLUMNS}
            row.update({key: float(raw[key]) for key in FLOAT_COLUMNS})
        except (ValueError, TypeError) as exc:
            raise BenchmarkError("malformed numeric field") from exc
        if any(not math.isfinite(row[key]) for key in FLOAT_COLUMNS):
            raise BenchmarkError("nonfinite benchmark metric")
        k, n, pct = row["K"], row["N"], row["target_zero_pct"]
        key = (k, n, pct)
        if key in seen:
            raise BenchmarkError("duplicate shape / zero-rate record")
        seen.add(key)
        if k <= 0 or n <= 0 or pct < 0 or pct > 100:
            raise BenchmarkError("invalid benchmark dimensions / zero rate")
        if not (0 <= row["actual_zero_pct"] <= 100):
            raise BenchmarkError("invalid observed zero rate")
        for name in ("scalar_us", "acfq_dense_us", "acfq_exact_zero_us",
                     "dense_vs_scalar", "exact_zero_vs_scalar"):
            if row[name] <= 0:
                raise BenchmarkError("timings and speed ratios must be positive")
        for name in ("dense_abs_error", "exact_zero_abs_error"):
            if row[name] < 0:
                raise BenchmarkError("absolute errors cannot be negative")
        if row["dense_neon_blocks"] < 0 or row["exact_zero_skipped_products"] < 0:
            raise BenchmarkError("negative operation counter")
        # The native binary logs printed median timings to 3dp and ratios to 4dp.
        for reported, elapsed in (
            ("dense_vs_scalar", "acfq_dense_us"),
            ("exact_zero_vs_scalar", "acfq_exact_zero_us"),
        ):
            ratio = row["scalar_us"] / row[elapsed]
            # Allows rounding of both timing columns (0.0005us each).
            tolerance = 0.00005 + 0.0005 / row[elapsed] * (1 + ratio)
            if abs(row[reported] - ratio) > tolerance:
                raise BenchmarkError("speedup does not match measured times")
        # With one dense NEON block = 32 MAC products on AArch64.
        # A zero-point-skipped product is one x==zero_point input by one row.
        if row["dense_neon_blocks"] > n * (k // 32):
            raise BenchmarkError("impossible NEON block count")
        skipped = row["exact_zero_skipped_products"]
        if skipped > k * n or skipped % n:
            raise BenchmarkError("invalid skipped-products count")
        zero_pct_from_counter = 100 * (skipped // n) / k
        if abs(row["actual_zero_pct"] - zero_pct_from_counter) > 0.0051:
            raise BenchmarkError("observed zero percentage contradicts counter")
        results.append(row)
    if not results:
        raise BenchmarkError("no benchmark rows found")
    return results


def check_committed_observation(text):
    rows = check_report(text)
    expected = {(1536, n, z) for n in (256, 2048) for z in (0, 25, 50)}
    observed = {(r["K"], r["N"], r["target_zero_pct"]) for r in rows}
    if observed != expected:
        raise BenchmarkError("expected six Gemma-shaped benchmark cases")
    for row in rows:
        if row["dense_neon_blocks"] != row["N"] * (row["K"] // 32):
            raise BenchmarkError("observed benchmark must report full dense NEON coverage")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    try:
        rows = check_committed_observation(args.csv.read_text(encoding="utf-8"))
    except (OSError, BenchmarkError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    dense = [r["dense_vs_scalar"] for r in rows]
    slower_sparse = sum(r["acfq_exact_zero_us"] > r["acfq_dense_us"]
                        for r in rows)
    print(f"PASS: {len(rows)} records; 2 shapes; 3 zero-rate conditions.")
    print(f"Observed dense vs independent scalar: {min(dense):.4f}x–{max(dense):.4f}x")
    print(f"Sparse exact-zero path slower than dense NEON: {slower_sparse}/{len(rows)} cases.")
    print("No measurement of LiteRT, actual Gemma token throughput or ACFQ runtime integration.")


if __name__ == "__main__":
    main()
