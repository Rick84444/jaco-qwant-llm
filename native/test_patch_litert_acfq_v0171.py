"""Regression coverage for the pinned LiteRT-LM 0.17.1 source adapter."""
from pathlib import Path
import importlib.util
import unittest

path = Path(__file__).with_name("patch_litert_acfq_v0171.py")
spec = importlib.util.spec_from_file_location("patch_0171", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

SOURCE = """
#include <cstdio>
resolver->AddCustom("x", nullptr);
interp_->ModifyGraphWithDelegate();
  int current_undelegated_nodes = 25;
      current_undelegated_nodes = counts.undelegated_nodes;
  compiled_model->CheckCpuTensors();
  tflite::InterpreterOptions interpreter_options;
"""


class Test0171Adapter(unittest.TestCase):
    def test_patches_source_with_correct_cpp_escapes(self):
        new, changed = module.patch_source(SOURCE)
        self.assertTrue(changed)
        self.assertIn(module.MARKER, new)
        self.assertIn(module.TRACE_MARKER, new)
        self.assertIn("jaco_acfq::ShadowRegistration()", new)
        self.assertIn("current_undelegated_nodes,", new)
        self.assertIn("gpu_partitions=%d npu_partitions=%d\\n", new)
        self.assertIn("jaco_trace[1]=='\\0'", new)
        self.assertNotIn("jaco_trace[1]=='\\\\0'", new)
        self.assertLess(new.index("registration_installed=1"),
                        new.index("tflite::InterpreterOptions"))
        self.assertEqual(module.patch_source(new), (new, False))

    def test_rejects_newer_layout(self):
        wrong = SOURCE.replace(
            "      current_undelegated_nodes = counts.undelegated_nodes;",
            "      prev_counts = counts;"
        )
        with self.assertRaises(module.PatchError):
            module.patch_source(wrong)

    def test_refuses_partial_patching(self):
        with self.assertRaises(module.PatchError):
            module.patch_source(SOURCE + module.MARKER)

    def test_missing_anchors_refused(self):
        for anchor in (
            module.ENTRY, module.DELEGATE_ANCHOR, module.END_ANCHOR,
            "int current_undelegated_nodes =",
        ):
            with self.subTest(anchor=anchor), self.assertRaises(module.PatchError):
                module.patch_source(SOURCE.replace(anchor, ""))


if __name__ == "__main__":
    unittest.main()
