"""Independent packed Q4/Q8 tests against actual native ACFQ kernel; not Gemma."""
import ctypes, math, pathlib, struct, subprocess, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
K,N=5,32
def pack(weights,fmt):
 out=bytearray()
 for row in weights:
  s=(max(abs(v) for v in row)/127 if fmt==1 else -max(row,key=abs)/8) or 1.
  s=struct.unpack('<e',struct.pack('<e',s))[0]
  out.extend(struct.pack('<e',s))
  if fmt==1:
   q=[max(-127,min(127,round(v/s))) for v in row]
   out.extend(struct.pack('32b',*q))
  else:
   q=[max(0,min(15,round(v/s)+8)) for v in row]
   out.extend(bytes(q[i]|(q[i+16]<<4) for i in range(16)))
 return bytes(out)
def unpack(data,fmt):
 rb=34 if fmt==1 else 18
 rows=[]
 for i in range(K):
  raw=data[i*rb:(i+1)*rb]
  s=struct.unpack('<e',raw[:2])[0]
  q=list(struct.unpack('32b',raw[2:])) if fmt==1 else [((b&15)-8) for b in raw[2:]]+[((b>>4)-8) for b in raw[2:]]
  rows.append([s*v for v in q])
 return rows
class QuantizedBridgeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory()
  target=str(pathlib.Path(cls.tmp.name)/'acfq.so')
  subprocess.run(['cc','-std=c11','-O2','-shared','-fPIC',str(ROOT/'acfq_kernel.c'),str(ROOT/'native/acfq_backend.c'),'-lm','-o',target],check=True)
  lib=ctypes.CDLL(target)
  cls.fn=lib.jaco_acfq_execute
  cls.fn.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int),ctypes.c_size_t,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t]
  cls.fn.restype=ctypes.c_int
  cls.lib=lib
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def verify(self,fmt,indices):
  w=[[0.05*math.sin(i*7+j/3)+0.01*math.cos(i+j) for j in range(N)] for i in range(K)]
  raw=pack(w,fmt)
  packed=(ctypes.c_ubyte*len(raw)).from_buffer_copy(raw)
  xvals=[.25,-.5,.75,0,1.25]
  x=(ctypes.c_float*K)(*xvals)
  y=(ctypes.c_float*N)()
  idx=(ctypes.c_int*len(indices))(*indices) if indices is not None else None
  rc=self.fn(fmt,packed,len(raw),K,N,x,K,idx,len(indices) if indices is not None else 0,y,N)
  self.assertEqual(rc,0)
  dw=unpack(raw,fmt)
  for j in range(N):
   ref=sum(xvals[i]*dw[i][j] for i in (range(K) if indices is None else indices))
   self.assertAlmostEqual(y[j],ref,delta=2e-6)
 def test_q4_dense(self):self.verify(2,None)
 def test_q4_sparse(self):self.verify(2,[0,2,4])
 def test_q8_dense(self):self.verify(1,None)
 def test_q8_sparse(self):self.verify(1,[0,2,4])
 def test_zero_sparse(self):self.verify(2,[])
if __name__=='__main__':unittest.main()
