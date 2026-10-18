"""Evidence contract regression tests, independent of local .litertlm."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "native/decode_evidence.py"
spec = importlib.util.spec_from_file_location("decode_evidence",path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def fixture():
    return json.loads((ROOT / "evidence/gemma4-e2b-decode-observed.json").read_text())

def report_from(snapshot):
    return {
        "decode_subgraphs":[{"dtype_pairs": snapshot["dtype_pairs"],
                             "fully_connected_count":snapshot["fully_connected_operators"]}],
        "all_decode_fc_examined":snapshot["fully_connected_operators"],
        "actual_gemma_acfq_execution":False
    }

class EvidenceTests(unittest.TestCase):
    def test_committed_snapshot_is_consistent(self):
        self.assertTrue(mod.validate_snapshot(fixture()))
    def test_local_report_matches_observed_counts(self):
        s=fixture()
        self.assertEqual(mod.compare(report_from(s),s)["fully_connected_operators"],277)
    def test_mismatched_report_fails_closed(self):
        s=fixture()
        r=report_from(s)
        r["decode_subgraphs"][0]["dtype_pairs"] = {"INT8 x INT4":277}
        with self.assertRaises(mod.EvidenceError):
            mod.compare(r,s)
    def test_inconsistent_snapshot_rejected(self):
        s=fixture()
        s["dtype_pairs"]["INT8 x INT4"] += 1
        with self.assertRaises(mod.EvidenceError):
            mod.validate_snapshot(s)
    def test_false_inference_claim_is_rejected(self):
        s=fixture()
        r=report_from(s)
        r["actual_gemma_acfq_execution"]=True
        with self.assertRaises(mod.EvidenceError):
            mod.compare(r,s)
if __name__ == "__main__":
    unittest.main()
