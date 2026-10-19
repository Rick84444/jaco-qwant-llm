# Termux LiteRT-LM binary-provenance inspection (read-only)

The isolated ACFQ LiteRT library **built successfully on GitHub's Linux
runner**, but the Android LiteRT-LM CLI may use a different pinned LiteRT
revision. The running Gemma service is NOT automatically linked against
the isolated build.

Official LiteRT-LM tags in GitHub:
- `v0.17.1` WORKSPACE -> LiteRT `9fe5be45564c868408e6514c8aabb83e211a0911`
- `v0.18.0` WORKSPACE -> LiteRT `26895c9fbcc25c43faa8c1a98cd1fd28951602c3`
- Our isolated LiteRT proof build -> `44455548be35bde21737b8a28ef0d027632c282d`

The running service's actual binary revision remains unverified even
when Python package metadata indicates one of these versions. Both older
LiteRT revisions lack the exact per-delegate source anchor
`prev_counts = counts;` required by our October patch; rebuilding
our October source and swapping a library blindly is unsafe.

## Run

```bash
cd ~/jaco-qwant-llm
git pull --ff-only origin main
python3 native/probe_litert_lm_android.py
```

This prints bounded JSON showing installed release metadata, CLI binary
type (ELF versus launcher script), ELF library dependencies if available,
Android/Termux toolchain presence, disk space, and model ID via HTTP GET
to localhost:9379/v1/models. It does **not** inspect weights, execute
inferencing, install/upgrade anything, stop processes, send to the
internet, or access user credentials. The user decides whether to share
the JSON.

## Follow-up decision

1. Resolve the actual LiteRT-LM release / executable provenance.
2. Use exact version-pinned LiteRT with a correspondingly adapted shadow
   patch and independent source/ABI CI checks.
3. Compile a separate test runtime/binary and verify the
   `builtin_callback_invoked` / `native_invocations` logs with a
   *genuine Gemma decode* (not just `/v1/models`).
4. Compare actual output parity and tokens/s before any deployment.

The existing compiled Linux `liblitert_runtime.a` and the native
ARM64 smoke tests are insufficient evidence of runtime integration.
