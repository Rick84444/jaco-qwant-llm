"""Source-patch regression tests: graph-delegate observability is opt-in,
non-invasive, and correctly emits C++ escapes.
"""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).with_name("patch_litert_acfq.py")
spec = importlib.util.spec_from_file_location("patch_litert", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

FAKE_UPSTREAM = '''
#include <cstdio>
resolver->AddCustom
ModifyGraphWithDelegate
      prev_counts = counts;
  compiled_model->CheckCpuTensors();
  tflite::InterpreterOptions interpreter_options;
'''


class TestDelegatePatch(unittest.TestCase):
    def test_inserts_trace_with_correct_cpp_escapes(self):
        result, changed = module.apply_patch(FAKE_UPSTREAM)
        self.assertTrue(changed)
        self.assertIn("JACO_ACFQ_DELEGATE_TRACE_V1", result)
        self.assertIn("delegate_applied=%s", result)
        self.assertIn("delegation_complete=1", result)
        # One backslash + 0 is the C++ NUL character literal.
        self.assertIn("jaco_trace[1]=='\\0'", result)
        self.assertIn("jaco_trace_done[1]=='\\0'", result)
        # Two backslashes in emitted C++ is a multichar constant, NOT NUL.
        self.assertNotIn("jaco_trace[1]=='\\\\0'", result)
        self.assertNotIn("jaco_trace_done[1]=='\\\\0'", result)
        self.assertIn("remaining_nodes=%d\\n", result)
        self.assertNotIn("remaining_nodes=%d\\\\n", result)
        self.assertLess(result.index("prev_counts = counts;"),
                        result.index("delegate_applied=%s"))
        self.assertLess(result.index("delegation_complete=1"),
                        result.index("compiled_model->CheckCpuTensors();"))
        again, changed_again = module.apply_patch(result)
        self.assertFalse(changed_again)
        self.assertEqual(again, result)

    def test_rejects_unknown_upstream_layout(self):
        with self.assertRaises(module.PatchError):
            module.apply_patch(FAKE_UPSTREAM.replace("prev_counts = counts;", ""))
        with self.assertRaises(module.PatchError):
            module.apply_patch(FAKE_UPSTREAM.replace(
                "compiled_model->CheckCpuTensors();", ""))


if __name__ == "__main__":
    unittest.main()
