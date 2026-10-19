"""Read-only Android installed-runtime probe regressions."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest import mock

FILE = Path(__file__).with_name("probe_litert_lm_android.py")
spec = importlib.util.spec_from_file_location("litert_probe", FILE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class TestProbe(unittest.TestCase):
    def test_release_pins_are_different_and_not_binary_proof(self):
        self.assertEqual(probe.version_pin("0.17.1")["revision"],
                         "9fe5be45564c868408e6514c8aabb83e211a0911")
        self.assertEqual(probe.version_pin("0.18.0")["revision"],
                         "26895c9fbcc25c43faa8c1a98cd1fd28951602c3")
        self.assertEqual(probe.version_pin("0.18.0")["relationship"],
                         "DIFFERENT_FROM_JACO_PIN")
        self.assertIsNone(probe.version_pin("0.21")["revision"])

    def test_deny_external_model_urls(self):
        for url in ("https://example.com/v1/models",
                    "http://example.com/v1/models",
                    "http://127.0.0.1:9379/secret",
                    "http://127.0.0.1:9379/v1/models?token=hello"):
            with self.assertRaises(ValueError):
                probe.read_local_models(url)

    def test_executable_kind_reads_only_four_bytes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "big_elf"
            with p.open("wb") as fh:
                fh.write(b"\\x7fELF")
                fh.write(b"X" * 65536)
            self.assertEqual(probe.executable_kind(str(p)), "ELF")
            self.assertEqual(probe.executable_kind(None), "absent")

    def test_only_model_id_is_read(self):
        payload = json.dumps({
            "data": [
                {"id": "gemma4-e2b", "secret": "NOT_SHARED"},
                {"id": "other", "model_path": "/private"},
            ]
        })
        response = io.BytesIO(payload.encode())
        with mock.patch.object(probe.urllib.request, "urlopen",
                               return_value=response):
            result = probe.read_local_models("http://127.0.0.1:9379/v1/models")
        self.assertEqual(result["model_ids"], ["gemma4-e2b", "other"])
        self.assertNotIn("NOT_SHARED", str(result))
        self.assertNotIn("/private", str(result))


if __name__ == "__main__":
    unittest.main()
