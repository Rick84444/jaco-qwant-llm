"""End-to-end checked C ABI tests for signed-packed Gemma INT8xINT4 math."""
import ctypes
import math
import pathlib
import random
import subprocess
import tempfile
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
class Params(ctypes.Structure):
    _fields_=[
        ("input",ctypes.POINTER(ctypes.c_int8)),("input_len",ctypes.c_size_t),
        ("input_scale",ctypes.c_float),("input_zero_point",ctypes.c_int32),
        ("weights",ctypes.POINTER(ctypes.c_uint8)),("weights_len",ctypes.c_size_t),
        ("k",ctypes.c_int),("n",ctypes.c_int),
        ("weight_scales",ctypes.POINTER(ctypes.c_float)),("weight_scales_len",ctypes.c_size_t),
        ("weight_zero_points",ctypes.POINTER(ctypes.c_int32)),("weight_zero_points_len",ctypes.c_size_t),
        ("bias",ctypes.POINTER(ctypes.c_float)),("bias_len",ctypes.c_size_t),
        ("output",ctypes.POINTER(ctypes.c_float)),("output_len",ctypes.c_size_t),
        ("skip_exact_zero",ctypes.c_int),
        ("executed_products",ctypes.POINTER(ctypes.c_uint64)),
        ("skipped_products",ctypes.POINTER(ctypes.c_uint64)),
        ("neon_blocks",ctypes.POINTER(ctypes.c_uint64))
    ]

def packed(weights):
    values=[v & 15 for row in weights for v in row]
    return bytes((values[i] | ((values[i+1] if i+1<len(values) else 0)<<4))
                for i in range(0,len(values),2))

class NativeGemmaInt4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        lib=str(pathlib.Path(cls.tmp.name)/"libgemma-int4.so")
        subprocess.run(["cc","-std=c11","-O2","-shared","-fPIC",
                        str(ROOT/"native/gemma_int4_kernel.c"),
                        "-lm","-o",lib],check=True)
        cls.native=ctypes.CDLL(lib)
        cls.native.jaco_gemma_int4_gemv.argtypes=[ctypes.POINTER(Params)]
        cls.native.jaco_gemma_int4_gemv.restype=ctypes.c_int
        cls.native.jaco_gemma_int4_required_bytes.argtypes=[ctypes.c_int,ctypes.c_int]
        cls.native.jaco_gemma_int4_required_bytes.restype=ctypes.c_size_t
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def invoke(self,k,n,seed,skip,zps=True,biases=True):
        rng=random.Random(seed)
        azp=-7
        x_values=[rng.randint(-128,127) for _ in range(k)]
        for j in range(0,k,7):x_values[j]=azp
        w=[[rng.randint(-8,7) for _ in range(k)] for _ in range(n)]
        wzp=[rng.randint(-8,7) if zps else 0 for _ in range(n)]
        scales=[rng.uniform(.001,.1) for _ in range(n)]
        bias=[rng.uniform(-.5,.5) if biases else 0. for _ in range(n)]
        blob=packed(w)
        aa=(ctypes.c_int8*k)(*x_values)
        ww=(ctypes.c_uint8*len(blob)).from_buffer_copy(blob)
        ss=(ctypes.c_float*n)(*scales)
        zz=(ctypes.c_int32*n)(*wzp)
        bb=(ctypes.c_float*n)(*bias)
        yy=(ctypes.c_float*n)(*([77.]*n))
        ops=ctypes.c_uint64();skipped=ctypes.c_uint64();neon=ctypes.c_uint64()
        p=Params(aa,k,.125,azp,ww,len(blob),k,n,ss,n,
                 zz if zps else None,n if zps else 0,
                 bb if biases else None,n if biases else 0,
                 yy,n,int(skip),ctypes.pointer(ops),ctypes.pointer(skipped),ctypes.pointer(neon))
        rc=self.native.jaco_gemma_int4_gemv(ctypes.byref(p))
        self.assertEqual(rc,0)
        for row in range(n):
            ref=.125*scales[row]*sum((x_values[j]-azp)*(w[row][j]-wzp[row])
                                     for j in range(k))+bias[row]
            self.assertAlmostEqual(yy[row],ref,delta=max(1e-4,abs(ref)*1e-6))
        self.assertEqual(ops.value+skipped.value,n*k)
        if skip:
            self.assertGreater(skipped.value,0)
        else:self.assertEqual(skipped.value,0)
        return p,yy
    def test_odd_k_cross_row_packing(self):self.invoke(33,3,3,False)
    def test_odd_k_exact_zero_skipping(self):self.invoke(33,3,3,True)
    def test_gemma_shape_1536_x_256_dense(self):self.invoke(1536,256,2,False)
    def test_gemma_shape_1536_x_256_sparse(self):self.invoke(1536,256,2,True)
    def test_nullable_metadata(self):self.invoke(32,2,5,True,False,False)
    def test_required_bytes(self):
        self.assertEqual(self.native.jaco_gemma_int4_required_bytes(33,3),50)
        self.assertEqual(self.native.jaco_gemma_int4_required_bytes(-1,3),0)
    def test_truncated_weight_buffer_rejected(self):
        p,yy=self.invoke(33,3,3,False)
        p.weights_len-=1
        before=list(yy)
        self.assertEqual(self.native.jaco_gemma_int4_gemv(ctypes.byref(p)),-2)
        self.assertEqual(list(yy),before)
    def test_missing_scales_rejected(self):
        p,yy=self.invoke(33,3,3,False)
        p.weight_scales_len-=1
        self.assertEqual(self.native.jaco_gemma_int4_gemv(ctypes.byref(p)),-2)
    def test_invalid_input_scale_rejected(self):
        p,yy=self.invoke(33,3,3,False)
        p.input_scale=0.0
        self.assertEqual(self.native.jaco_gemma_int4_gemv(ctypes.byref(p)),-1)
    def test_invalid_weight_zero_point_rejected(self):
        p,yy=self.invoke(33,3,3,False)
        p.weight_zero_points[0]=9
        self.assertEqual(self.native.jaco_gemma_int4_gemv(ctypes.byref(p)),-1)
if __name__=="__main__":unittest.main()
