"""Verify an existing pinned LiteRT checkout can be upgraded safely."""
import unittest

import patch_litert_acfq as patch
from refresh_litert_acfq_source import classify_source, expected_legacy_source


UPSTREAM = '''
#include <cstdio>
resolver->AddCustom("test", nullptr);
interp_->ModifyGraphWithDelegate();
      prev_counts = counts;
  compiled_model->CheckCpuTensors();
  tflite::InterpreterOptions interpreter_options;
'''


class TestSparseSourceRefresh(unittest.TestCase):
    def test_legacy_hook_is_reconstructed_exactly(self):
        old = expected_legacy_source(UPSTREAM)
        self.assertIn("JACO_ACFQ_SHADOW_SOURCE_PATCH_V1", old)
        self.assertNotIn("JACO_ACFQ_DELEGATE_TRACE_V1", old)
        new, status = classify_source(UPSTREAM, old)
        self.assertEqual(status, "verified-legacy")
        self.assertIn("JACO_ACFQ_DELEGATE_TRACE_V1", new)
        self.assertIn("delegate_applied=%s", new)
        self.assertIn("delegation_complete=1", new)

    def test_current_source_does_not_change(self):
        new = patch.apply_patch(UPSTREAM)[0]
        expected, status = classify_source(UPSTREAM, new)
        self.assertEqual(status, "already-current")
        self.assertEqual(expected, new)

    def test_manual_source_edits_are_not_lost(self):
        old = expected_legacy_source(UPSTREAM)
        with self.assertRaisesRegex(ValueError, "unexpected edits"):
            classify_source(UPSTREAM, old + "\n// user-owned custom fix\n")
        with self.assertRaisesRegex(ValueError, "unexpected edits"):
            classify_source(UPSTREAM, UPSTREAM)


if __name__ == "__main__":
    unittest.main()
