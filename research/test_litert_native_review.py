import importlib.util
from pathlib import Path
import unittest
file=Path(__file__).with_name("litert_native_review.py")
spec=importlib.util.spec_from_file_location("native_review",file)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class NativeReviewTests(unittest.TestCase):
 def test_no_evidence_blocks(self):
  r=mod.report({},"a"*40)
  self.assertEqual(r["review"]["decision"],"BLOCK_NATIVE_ACFQ")
  self.assertEqual(len(r["sources"]),len(mod.TARGETS))
  self.assertTrue(all(x["status"]=="not-retrieved" for x in r["sources"]))
 def test_evidence_is_only_candidate(self):
  src={"executor_settings":"struct CpuConfig { bool enable_ynnpack; Backend::CPU; };"}
  r=mod.report(src,"a"*40)
  self.assertEqual(r["review"]["actions"][0]["proof_level"],"lexical")
  self.assertFalse(r["inference_changed"])
if __name__=="__main__":unittest.main()
