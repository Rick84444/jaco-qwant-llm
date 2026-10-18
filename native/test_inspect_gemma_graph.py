import importlib.util
import pathlib
import struct
import tempfile
import unittest

p=pathlib.Path(__file__).with_name("inspect_gemma_graph.py")
s=importlib.util.spec_from_file_location("graph",p)
g=importlib.util.module_from_spec(s);s.loader.exec_module(g)

def fixture(bad_range=False):
    """Synthetic LiteRT-LM FlatBuffer with one tiny TFL3 submodel."""
    b=bytearray(320)
    b[:8]=b"LITERTLM"
    struct.pack_into("<III",b,8,1,0,0)
    struct.pack_into("<Q",b,24,224)
    struct.pack_into("<I",b,32,32)     # root at 64
    struct.pack_into("<HHHH",b,56,8,12,0,8)
    struct.pack_into("<i",b,64,8)
    struct.pack_into("<I",b,72,24)     # SectionMetadata at 96
    struct.pack_into("<HHH",b,88,6,8,4)
    struct.pack_into("<i",b,96,8)
    struct.pack_into("<I",b,100,12)    # sections vector at 112
    struct.pack_into("<I",b,112,1)
    struct.pack_into("<I",b,116,44)    # SectionObject at 160
    struct.pack_into("<HHHHHH",b,144,12,28,0,8,16,24)
    struct.pack_into("<i",b,160,16)
    struct.pack_into("<Q",b,168,256)
    struct.pack_into("<Q",b,176,400 if bad_range else 300)
    struct.pack_into("<B",b,184,3)   # TFLiteModel
    struct.pack_into("<I",b,256,20)
    b[260:264]=b"TFL3"
    struct.pack_into("<HHH",b,268,6,8,4)
    struct.pack_into("<i",b,276,8)
    struct.pack_into("<I",b,280,3)
    return b

class FlatbufferInventoryTests(unittest.TestCase):
    def test_valid_container_and_tflite_submodel(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/"sample.litertlm"
            path.write_bytes(fixture())
            result=g.inspect_file(path)
            self.assertEqual(result["sections"][0]["kind"],"TFLiteModel")
            self.assertEqual(result["sections"][0]["bytes"],44)
            self.assertEqual(result["sections"][0]["graph"]["schema_version"],3)
            self.assertEqual(result["sections"][0]["graph"]["status"],"tflite-schema-inventory")
            self.assertFalse(result["acfq_inference_integrated"])
    def test_invalid_section_extent_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/"bad.litertlm"
            path.write_bytes(fixture(True))
            with self.assertRaises(g.FormatError):g.inspect_file(path)
    def test_wrong_magic(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/"bad.litertlm"
            path.write_bytes(b"NOFORMAT"+bytes(48))
            with self.assertRaises(g.FormatError):g.inspect_file(path)
    def test_short_file(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/"short"
            path.write_bytes(b"LITERTLM")
            with self.assertRaises(g.FormatError):g.inspect_file(path)
    def test_bad_header_end(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/"bad"
            path.write_bytes(b"LITERTLM"+bytes(56))
            with self.assertRaises(g.FormatError):g.inspect_file(path)
    def test_wrong_tflite_identifier(self):
        self.assertEqual(g.inspect_tflite(bytearray(40),0,40)["status"],
                         "unsupported-tflite-identifier")
    def test_bounds(self):
        reader=g.FB(bytearray(40),8,32)
        with self.assertRaises(g.FormatError):reader.u32(30)
if __name__=="__main__":unittest.main()
