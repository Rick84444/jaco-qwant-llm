"""Run a real ACFQ GEMV through a checked C ABI, not through LiteRT."""
import ctypes
import pathlib
import subprocess
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
class BridgeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory()
  target=pathlib.Path(cls.temp.name)/"libjaco-acfq-test.so"
  subprocess.run(["cc","-std=c11","-O2","-fPIC","-shared",
                  str(ROOT/"acfq_kernel.c"),str(ROOT/"native/acfq_backend.c"),
                  "-lm","-o",str(target)],check=True)
  cls.lib=ctypes.CDLL(str(target))
  cls.lib.jaco_acfq_execute.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,
      ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,
      ctypes.POINTER(ctypes.c_int),ctypes.c_size_t,
      ctypes.POINTER(ctypes.c_float),ctypes.c_size_t]
  cls.lib.jaco_acfq_execute.restype=ctypes.c_int
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def run_case(self,idx=None):
  k,n=4,32
  vals=[float((i+1)*(j-8))/128 for i in range(k) for j in range(n)]
  w=(ctypes.c_float*(k*n))(*vals)
  x=(ctypes.c_float*k)(.25,-.5,.75,1.0)
  y=(ctypes.c_float*n)()
  indices=(ctypes.c_int*len(idx))(*idx) if idx is not None else None
  result=self.lib.jaco_acfq_execute(0,w,ctypes.sizeof(w),k,n,x,k,indices,
                             len(idx) if idx is not None else 0,y,n)
  return result,list(y),vals,list(x)
 def test_dense_exact(self):
  code,out,weights,x=self.run_case()
  self.assertEqual(code,0)
  for j,v in enumerate(out):
   ref=sum(x[i]*weights[i*32+j] for i in range(4))
   self.assertAlmostEqual(v,ref,places=6)
 def test_sparse_exact_for_active_indices(self):
  code,out,weights,x=self.run_case([0,2])
  self.assertEqual(code,0)
  for j,v in enumerate(out):
   ref=sum(x[i]*weights[i*32+j] for i in (0,2))
   self.assertAlmostEqual(v,ref,places=6)
 def test_invalid_index_rejected(self):
  code,_,_,_=self.run_case([2,0])
  self.assertEqual(code,-1)
 def test_invalid_weight_extent_rejected(self):
  w=(ctypes.c_float*32)()
  x=(ctypes.c_float*4)()
  y=(ctypes.c_float*32)()
  code=self.lib.jaco_acfq_execute(0,w,ctypes.sizeof(w),4,32,x,4,None,0,y,32)
  self.assertEqual(code,-2)
if __name__=="__main__":unittest.main()
