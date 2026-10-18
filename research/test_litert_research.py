import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).with_name("litert_research.py")
spec=importlib.util.spec_from_file_location("litert_research",p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ReviewTests(unittest.TestCase):
 def test_no_evidence_is_blocking(self):
  report=m.assess({},"a"*40)
  self.assertEqual(report["senior_review"]["decision"],"BLOCK_AFCQ_INFERENCE")
  self.assertFalse(report["acfq_inference_enabled"])
  self.assertGreater(len(report["senior_review"]["findings"]),len(m.GATES))
 def test_match_does_not_prove_acceleration(self):
  data={k:" ".join(v.values()) for k,v in m.CHECKS.items()}
  report=m.assess(data,"b"*40)
  self.assertTrue(all(i["found"] for i in report["evidence"]))
  self.assertEqual(report["senior_review"]["decision"],"BLOCK_AFCQ_INFERENCE")
 def test_unpinned_revision_rejected(self):
  with self.assertRaises(ValueError):m.retrieve("main")
if __name__=="__main__":unittest.main()
