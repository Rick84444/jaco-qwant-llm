"""Isolated LiteRT-LM ARM64 staging safety tests."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

file = Path(__file__).with_name("stage_litert_lm_arm64.py")
spec = importlib.util.spec_from_file_location("acfq_stage", file)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def elf():
    head = bytearray(64)
    head[:6] = b"\x7fELF\x02\x01"
    head[16:18] = (3).to_bytes(2, "little")
    head[18:20] = (183).to_bytes(2, "little")
    return bytes(head)


class TestStaging(unittest.TestCase):
    def test_validation_rejects_non_arm64(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bin"
            path.write_bytes(b"NOT ELF")
            with self.assertRaises(ValueError):
                stage.validate_arm64_elf(path)

    def test_digest_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            file = root / stage.DSO
            file.write_bytes(elf() + b"test")
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            (root / "SHA256SUMS").write_text(digest + "  " + file.name + "\n")
            _, observed = stage.validate_artifact(root)
            self.assertEqual(observed, digest)
            (root / "SHA256SUMS").write_text("0" * 64 + "  " + file.name)
            with self.assertRaisesRegex(ValueError, "checksum"):
                stage.validate_artifact(root)

    def test_private_staging_preserves_original_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            package = home / "installed" / "litert_lm"
            package.mkdir(parents=True)
            original = package / "liblitert-lm.so"
            original.write_bytes(b"PRODUCTION")
            (package / "_ffi.py").write_text("x=1\n")
            candidate = home / "experimental.so"
            candidate.write_bytes(elf() + b"PATCHED")
            target = home / stage.NAME
            with mock.patch.object(stage.Path, "home", return_value=home):
                result = stage.stage_private(package, candidate, target)
            self.assertEqual(result, target)
            self.assertEqual(original.read_bytes(), b"PRODUCTION")
            self.assertEqual((target / "litert_lm" / "liblitert-lm.so").read_bytes(),
                             candidate.read_bytes())
            self.assertFalse((target / "STAGE_MANIFEST.json").is_symlink())
            with mock.patch.object(stage.Path, "home", return_value=home):
                with self.assertRaises(FileExistsError):
                    stage.stage_private(package, candidate, target)

    def test_outside_home_refused(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            source = home / "wheel"
            source.mkdir()
            obj = home / "artifact"
            obj.write_bytes(elf())
            with mock.patch.object(stage.Path, "home", return_value=home):
                with self.assertRaisesRegex(ValueError, "exactly"):
                    stage.stage_private(source, obj, home / "wheel2")


if __name__ == "__main__":
    unittest.main()
