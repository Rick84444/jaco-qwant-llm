import importlib.util,pathlib,tempfile,unittest
p=pathlib.Path(__file__).with_name('gemma_install_probe.py')
s=importlib.util.spec_from_file_location('gemma_probe',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class ProbeTests(unittest.TestCase):
 def test_existing(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'data';p.write_bytes(b'hello')
   r=m.report(p,p)
   self.assertEqual(r['model']['size'],5)
   self.assertFalse(r['inference_integrated'])
 def test_missing(self):
  with tempfile.TemporaryDirectory() as d:
   r=m.report(pathlib.Path(d)/'no',pathlib.Path(d)/'no')
   self.assertFalse(r['model']['present'])
if __name__=='__main__':unittest.main()
