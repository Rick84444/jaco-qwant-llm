"""Fail-closed Bazel integration regression for LiteRT 0.17.1."""
import unittest
from pathlib import Path
import importlib.util

path = Path(__file__).with_name("prepare_litert_bazel_v0171.py")
spec = importlib.util.spec_from_file_location("jaco_bazel_prepare", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SOURCE = (
    '#include "litert/runtime/jaco_acfq_shadow.h"\n'
    'extern "C" {\n'
    '#include "litert/runtime/jaco_acfq_native/gemma_int4_kernel.c"\n'
    '}\n'
    '// JACO_ACFQ_SHADOW_SOURCE_PATCH_V1\n'
    '// JACO_ACFQ_TRACE_0171_V1\n'
    '// JACO_ACFQ_TRACE_0171_V1\n'
)
BUILD = mod.BAZEL_RULE_ANCHOR + '    deps = [],\n)\n'


class PrepareBazelTests(unittest.TestCase):
    def test_declares_native_file_once_and_removes_inline_c(self):
        cc, build = mod.prepare_bazel(SOURCE, BUILD)
        self.assertNotIn('extern "C" {', cc)
        self.assertNotIn('gemma_int4_kernel.c', cc)
        self.assertIn("jaco_acfq_shadow.h", cc)
        self.assertIn('"jaco_acfq_native/gemma_int4_kernel.c"', build)
        self.assertIn('"jaco_acfq_native/gemma_int4_kernel.h"', build)
        self.assertIn('"jaco_acfq_shadow.h"', build)
        self.assertEqual(build.count('"jaco_acfq_native/gemma_int4_kernel.c"'), 1)
        self.assertEqual(cc.count(mod.MARKER), 1)

    def test_refuses_unknown_build_layout(self):
        with self.assertRaises(mod.PatchError):
            mod.prepare_bazel(SOURCE, BUILD.replace('"compiled_model"', '"renamed"'))

    def test_refuses_missing_source_marker_or_double_patch(self):
        with self.assertRaises(mod.PatchError):
            mod.prepare_bazel(SOURCE.replace(mod.MARKER, ""), BUILD)
        cc, build = mod.prepare_bazel(SOURCE, BUILD)
        with self.assertRaises(mod.PatchError):
            mod.prepare_bazel(cc, build)

    def test_does_not_modify_build_when_source_patch_is_missing(self):
        with self.assertRaises(mod.PatchError):
            mod.prepare_bazel(SOURCE.replace("extern \"C\" {", "extern CUSTOM {"), BUILD)


if __name__ == "__main__":
    unittest.main()
