"""Unit tests for Gemma decode quantization evidence; no false integration."""
import importlib.util
import pathlib
import struct
import sys
import unittest

folder=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(folder))
spec=importlib.util.spec_from_file_location("gemma_fc",folder/"gemma_decode_fc.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class QuantizationEvidenceTests(unittest.TestCase):
    def test_q4_int8_must_not_be_declared_compatible(self):
        a={"dtype":"INT8","shape":[1,1,1536]}
        w={"dtype":"INT4","shape":[2048,1536]}
        self.assertEqual(mod.compatibility(a,w),"unsupported-direct-ACFQ-ABI")
    def test_f32_is_not_verified_compatible(self):
        a={"dtype":"FLOAT32","shape":[1,1,1536]}
        w={"dtype":"FLOAT32","shape":[2048,1536]}
        self.assertIn("needs",mod.compatibility(a,w))
    def test_quant_vector_preview(self):
        buf=bytearray(128)
        # vector starts at 48 (len=4, samples at 52), FlatBuffer table field 2
        struct.pack_into("<HHHHHH",buf,8,12,16,0,0,4,0)
        struct.pack_into("<i",buf,24,16)
        struct.pack_into("<I",buf,28,20)
        struct.pack_into("<I",buf,48,4)
        for i,v in enumerate((0.1,0.2,0.3,0.4)):
            struct.pack_into("<f",buf,52+4*i,v)
        reader=mod.FB(buf,0,len(buf))
        evidence=mod.vector_sample(reader,24,2,"<f",limit=2)
        self.assertEqual(evidence["count"],4)
        self.assertEqual(len(evidence["preview"]),2)
    def test_large_valid_quantization_vector_is_previewed(self):
        # Regression: a legitimate scale vector >250,000 elements must not
        # hit the structural-vector limit or be copied into Python RAM.
        n=250001
        buf=bytearray(56+4*n)
        struct.pack_into("<HHHHHH",buf,8,12,16,0,0,4,0)
        struct.pack_into("<i",buf,24,16)
        struct.pack_into("<I",buf,28,20)
        struct.pack_into("<I",buf,48,n)
        struct.pack_into("<f",buf,52,1.25)
        struct.pack_into("<f",buf,56,2.5)
        info=mod.vector_sample(mod.FB(buf,0,len(buf)),24,2,"<f")
        self.assertEqual(info["count"],n)
        self.assertEqual(info["preview"],[1.25,2.5,0.0])

    def test_oversized_out_of_bounds_quantization_vector_rejected(self):
        buf=bytearray(128)
        struct.pack_into("<HHHHHH",buf,8,12,16,0,0,4,0)
        struct.pack_into("<i",buf,24,16)
        struct.pack_into("<I",buf,28,20)
        struct.pack_into("<I",buf,48,300000)
        with self.assertRaises(mod.FormatError):
            mod.vector_sample(mod.FB(buf,0,len(buf)),24,2,"<f")

    def test_blockwise_quantization_details(self):
        buf=bytearray(128)
        # QuantizationParameters table 40 with fields 4 and 5.
        struct.pack_into("<HH",buf,8,16,16)
        struct.pack_into("<HH",buf,20,4,8)
        struct.pack_into("<i",buf,40,32)
        struct.pack_into("<B",buf,44,2)
        struct.pack_into("<I",buf,48,16)
        # BlockwiseQuantization table 64 with field indices 0,1,2.
        struct.pack_into("<HHHHH",buf,54,10,16,4,8,12)
        struct.pack_into("<i",buf,64,10)
        struct.pack_into("<iii",buf,68,13,-1,32)
        info=mod.quantization_details(mod.FB(buf,0,len(buf)),40)
        self.assertEqual(info["type"],"BlockwiseQuantization")
        self.assertEqual(info["scale_tensor_index"],13)
        self.assertEqual(info["zero_point_tensor_index"],-1)
        self.assertEqual(info["block_size"],32)

    def test_invalid_reference_is_rejected(self):
        buf=bytearray(80)
        struct.pack_into("<HHHHHH",buf,8,12,16,0,0,4,0)
        struct.pack_into("<i",buf,24,16)
        struct.pack_into("<I",buf,28,75)
        with self.assertRaises(mod.FormatError):
            mod.vector_sample(mod.FB(buf,0,len(buf)),24,2,"<f")
if __name__=="__main__":unittest.main()
