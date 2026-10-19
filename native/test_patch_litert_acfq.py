"""Source patch contract tests; these do not compile upstream LiteRT."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).with_name('patch_litert_acfq.py')
s=importlib.util.spec_from_file_location('litert_patch',path)
p=importlib.util.module_from_spec(s)
s.loader.exec_module(p)

class PatchSourceTests(unittest.TestCase):
    @staticmethod
    def fixture():
        return ('#include <cstdio>\n'
                '#include "tflite/delegates/xnnpack/xnnpack_delegate.h"\n'
                'resolver->AddCustom("test", nullptr);\n'
                '  tflite::InterpreterOptions interpreter_options;\n'
                'interp_->ModifyGraphWithDelegate();\n')
    def test_patch_is_before_interpreter_creation(self):
        result,changed=p.apply_patch(self.fixture())
        self.assertTrue(changed)
        self.assertIn('jaco_acfq::ShadowRegistration()',result)
        self.assertLess(result.index('registration_installed=1'),
                        result.index('tflite::InterpreterOptions'))
        self.assertIn('JACO_ACFQ_SHADOW',result)
        self.assertIn('extern "C"',result)
    def test_idempotent(self):
        once,changed=p.apply_patch(self.fixture())
        self.assertTrue(changed)
        twice,again=p.apply_patch(once)
        self.assertFalse(again)
        self.assertEqual(once,twice)
    def test_unrecognized_source_refused(self):
        with self.assertRaises(p.PatchError):
            p.apply_patch('#include <cstdio>\n')
    def test_multiple_anchors_refused(self):
        with self.assertRaises(p.PatchError):
            p.apply_patch(self.fixture()+
                '  tflite::InterpreterOptions interpreter_options;\n')
if __name__=="__main__":unittest.main()
