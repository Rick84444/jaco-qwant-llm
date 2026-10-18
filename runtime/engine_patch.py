"""Read-only LiteRT-LM/ACFQ compatibility probe.

Does NOT monkeypatch Engine, change inference or enable acceleration.
Run: python3 runtime/engine_patch.py
"""
from __future__ import annotations

import ctypes
import importlib.util
import json
from pathlib import Path

def inspect_environment(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parent.parent
    lib = root / "libacfq.so"
    try:
        litert = importlib.util.find_spec("litert_lm") is not None
    except (ModuleNotFoundError, ValueError):
        litert = False
    result = {
        "component": "jaco-acfq-engine-probe",
        "litert_python_module_present": litert,
        "acfq_library_present": lib.is_file(),
        "acfq_library_loadable": False,
        "engine_graph_hook_verified": False,
        "acfq_inference_integrated": False,
        "acceleration_enabled": False,
        "reason": "No verified LiteRT operator/delegate binding exists for the ACFQ kernel."
    }
    if lib.is_file():
        try:
            ctypes.CDLL(str(lib))
            result["acfq_library_loadable"] = True
        except OSError as exc:
            result["library_error"] = str(exc)
    return result

def main() -> None:
    print(json.dumps(inspect_environment(), indent=2))

if __name__ == "__main__":
    main()
