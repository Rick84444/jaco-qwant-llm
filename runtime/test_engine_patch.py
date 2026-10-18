import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).with_name("engine_patch.py")
spec = importlib.util.spec_from_file_location("engine_patch", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class EngineProbeTest(unittest.TestCase):
    def test_default_never_claims_acceleration(self):
        with tempfile.TemporaryDirectory() as p:
            result = module.inspect_environment(Path(p))
        self.assertFalse(result["acceleration_enabled"])
        self.assertFalse(result["engine_graph_hook_verified"])
        self.assertFalse(result["acfq_inference_integrated"])
        self.assertFalse(result["acfq_library_present"])

if __name__ == "__main__":
    unittest.main()
