#!/usr/bin/env python3
"""Validate observed Gemma decode FC findings against a reproducible local report.

No LiteRT integration is claimed; this checks report consistency and snapshot
agreement only. No model weights or personal paths are committed.
"""
import argparse
import json
from pathlib import Path

DEFAULT_SNAPSHOT = Path(__file__).resolve().parent.parent / "evidence/gemma4-e2b-decode-observed.json"

class EvidenceError(ValueError):
    pass

def validate_snapshot(snapshot):
    pairs = snapshot.get("dtype_pairs")
    total = snapshot.get("fully_connected_operators")
    if not isinstance(pairs, dict) or not isinstance(total, int) or total <= 0:
        raise EvidenceError("snapshot has invalid counts")
    if any(not isinstance(n, int) or n < 0 for n in pairs.values()):
        raise EvidenceError("snapshot has invalid dtype pair count")
    if sum(pairs.values()) != total:
        raise EvidenceError("snapshot dtype pair counts do not sum to total")
    seen = set()
    for op in snapshot.get("sampled_operators", []):
        idx = op.get("operator_index")
        shape = op.get("weight_shape")
        count = op.get("scale_count")
        zcount = op.get("zero_point_count")
        if not isinstance(idx, int) or idx < 0 or idx in seen:
            raise EvidenceError("invalid or repeated sampled operator index")
        seen.add(idx)
        if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(x,int) and x>0 for x in shape):
            raise EvidenceError("invalid sample weight shape")
        if count != shape[0] or zcount != shape[0]:
            raise EvidenceError("observed per-output-channel quantization dimensions mismatch")
        pair = f"{op.get('activation_dtype')} x {op.get('weight_dtype')}"
        if pairs.get(pair, 0) < 1:
            raise EvidenceError("sample dtype pair missing from histogram")
    return True

def normalize_report(report):
    graphs = report.get("decode_subgraphs")
    if not isinstance(graphs, list) or not graphs:
        raise EvidenceError("report has no decode subgraphs")
    counts = {}
    total = 0
    for graph in graphs:
        hist = graph.get("dtype_pairs", {})
        graph_total = graph.get("fully_connected_count")
        if not isinstance(hist, dict) or not isinstance(graph_total, int) or graph_total < 0:
            raise EvidenceError("invalid decode subgraph")
        if sum(hist.values()) != graph_total:
            raise EvidenceError("subgraph histogram total mismatch")
        total += graph_total
        for kind, n in hist.items():
            if not isinstance(kind, str) or not isinstance(n, int) or n < 0:
                raise EvidenceError("invalid dtype histogram")
            counts[kind] = counts.get(kind, 0) + n
    if report.get("all_decode_fc_examined") != total:
        raise EvidenceError("whole-report count mismatch")
    return {"fully_connected_operators": total, "dtype_pairs": counts, "decode_subgraphs": len(graphs)}

def compare(report, snapshot):
    validate_snapshot(snapshot)
    measured = normalize_report(report)
    if measured["fully_connected_operators"] != snapshot["fully_connected_operators"]:
        raise EvidenceError("FC operator count differs from observed snapshot")
    if measured["dtype_pairs"] != snapshot["dtype_pairs"]:
        raise EvidenceError("dtype pair histogram differs from observed snapshot")
    if measured["decode_subgraphs"] != snapshot["decode_subgraphs"]:
        raise EvidenceError("number of decode subgraphs differs")
    if report.get("actual_gemma_acfq_execution") is not False:
        raise EvidenceError("report cannot claim ACFQ execution")
    return measured

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("bench/gemma-decode-fc.json"))
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    snapshot = json.loads(args.snapshot.read_text())
    try:
        summary = compare(report, snapshot)
    except EvidenceError as e:
        parser.exit(1, f"FAIL: {e}\n")
    print("PASS: local decode operator histogram matches committed observation.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("NOTE: does not prove actual ACFQ integration, speedup or full weight compatibility.")

if __name__ == "__main__":
    main()
