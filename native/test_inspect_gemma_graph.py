import importlib.util
from pathlib import Path
import tempfile
import unittest
file=Path(__file__).with_name('inspect_gemma_graph.py')
spec=importlib.util.spec_from_file_location('inspect_gemma_graph',file)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class GraphInspectorTests(unittest.TestCase):
 def test_wrong_magic(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'bad.litertlm'
   path.write_bytes(b'NOFORMAT'+b'\\0'*48)
   with self.assertRaises(mod.FormatError):mod.inspect_file(path)
 def test_short_file(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'short'
   path.write_bytes(b'LITERTLM')
   with self.assertRaises(mod.FormatError):mod.inspect_file(path)
 def test_bad_header_end(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'bad'
   path.write_bytes(b'LITERTLM'+b'\\0'*56)
   with self.assertRaises(mod.FormatError):mod.inspect_file(path)
 def test_bad_tflite_id(self):
  raw=bytearray(40)
  f=mod.inspect_tflite(raw,0,len(raw))
  self.assertEqual(f['status'],'unsupported-tflite-identifier')
 def test_bounds(self):
  reader=mod.FB(bytearray(40),8,32)
  with self.assertRaises(mod.FormatError):reader.u32(30)
if __name__=='__main__':unittest.main()
