# LiteRT-LM architecture review: verifiable evidence gates

The goal is to study upstream Gemma execution architecture across four specialized
research roles: Python Engine, native C-FFI, backend selection, and Android builds.
A separate **deterministic senior-review gate** identifies missing evidence and
blocks native ACFQ enablement. These are analytical workstreams, **not launched
autonomous AI subagents**. Evidence matches are source locators, not proof.

Pinned source references from Google's public LiteRT-LM repository:

- `python/litert_lm/engine.py` — Engine settings and conversation.
- `python/litert_lm/_ffi.py` — ctypes/native library loading.
- `python/litert_lm/interfaces.py` — Backend CPU/GPU/NPU options.
- `docs/getting-started/build-and-run.md` — Bazel and Android build.

Run against a specific **40-character commit hash** from upstream:

```sh
python3 research/litert_research.py --revision <UPSTREAM_SHA> --fetch
python3 -m unittest discover -s research -p 'test_*.py'
```

For offline evaluation use `--source-dir /path/to/LiteRT-LM` instead of
`--fetch`. The report includes SHA256 source hashes and line-level evidence.
Upstream versions change; verify correspondence with the installed local
`liblitert-lm.so` before deriving implementation steps.

An optimization may ship only after identifying a supported native operator
registration mechanism, quantization formats, tensor layout, numerical fidelity,
whole-model quality, throughput improvement over the same model baseline and
a reversible opt-in. Python hooking around `create_conversation()` or merely
loading `libacfq.so` does not satisfy this. Gemma LiteRT on port 9379 and
JACO adapter on 3131 are never changed by this tool.
