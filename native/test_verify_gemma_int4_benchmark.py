"""Regression tests for logged on-device INT8xINT4 microbenchmark."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "native/verify_gemma_int4_benchmark.py"
spec = importlib.util.spec_from_file_location("verify_gemma_int4_benchmark", MODULE)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class BenchmarkEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "evidence/gemma-int4-pixel10pro-2026-10-19.csv").read_text()

    def test_committed_six_cases_pass(self):
        rows = check.check_committed_observation(self.source)
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["N"] for r in rows}, {256, 2048})

    def test_header_mismatch_rejected(self):
        with self.assertRaises(check.BenchmarkError):
            check.check_report(self.source.replace("scalar_us", "fake_time"))

    def test_modified_speed_ratio_rejected(self):
        corrupted = self.source.replace("8.8864,1.2214", "99.0000,1.2214")
        with self.assertRaises(check.BenchmarkError):
            check.check_report(corrupted)

    def test_modified_skip_counter_rejected(self):
        corrupted = self.source.replace(",12288,1024", ",12288,1025")
        with self.assertRaises(check.BenchmarkError):
            check.check_report(corrupted)

    def test_nonfinite_latency_rejected(self):
        corrupted = self.source.replace("971.953", "NaN")
        with self.assertRaises(check.BenchmarkError):
            check.check_report(corrupted)

    def test_missing_row_rejected(self):
        cut = self.source.replace(
            "1536,2048,50,50.20,2345.104,284.401,1689.922,8.2458,1.3877,0.00000000,0.00000000,98304,1579008\n",
            ""
        )
        with self.assertRaises(check.BenchmarkError):
            check.check_committed_observation(cut)

    def test_duplicate_case_rejected(self):
        added = self.source + self.source.splitlines()[-1] + "\n"
        with self.assertRaises(check.BenchmarkError):
            check.check_report(added)


if __name__ == "__main__":
    unittest.main()
