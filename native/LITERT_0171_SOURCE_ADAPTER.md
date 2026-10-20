# LiteRT-LM 0.17.1 — separately pinned source adapter

**Source-only prototype. Not an Android build, Gemma integration, or speedup.**

Pixel 10 Pro reports the running LiteRT-LM 0.17.1 wheel has a 37.44 MiB
liblitert-lm.so and dynamic exports including Register_FULLY_CONNECTED.
ELF FLAGS contain SYMBOLIC and BIND_NOW; readelf -Wr produced no matching
PLT relocations for the candidate functions. LD_PRELOAD is **not a reliable
route** to intercept this DSO's internal calls.

Official LiteRT-LM 0.17.1 WORKSPACE pins LiteRT Git revision
9fe5be45564c868408e6514c8aabb83e211a0911.
This is expected source dependency, not independently verified wheel fingerprint.

The October 19 LiteRT revision uses prev_counts = counts; while the
v0.17.1 dependency uses current_undelegated_nodes = counts.undelegated_nodes;
Therefore the current source patcher cannot be reused unchanged.

New native/patch_litert_acfq_v0171.py is fail-closed, opt-in, and only
accepts a separate ~/jaco-litert-acfq-0171-source checkout at the exact
expected SHA. It refuses modified source, existing backups and target files.

CI checks synthetic fixtures **and** the actual pinned upstream
compiled_model.cc source in memory. This does not compile the 0.17.1
runtime or prove that the callback will run in a Gemma decode.

Before building any isolated Android 0.17.1 LiteRT-LM executable:

1. Confirm ABI/header compatibility against this exact pinned LiteRT.
2. Link a separate binary to the compatible LiteRT-LM version.
3. Observe CPU registration, delegation, real model callback and ACFQ invocation.
4. Verify INT8xINT4 weight semantics, bias, outputs, and tokens/s.

Never overwrite the production wheel, model or port-9379 server.

## ABI test against 0.17.1's *actual* TFLite headers

The existing smoke test can now select the exact 0.17.1 header revision:

```bash
JACO_LITERT_ABI_PIN=9fe5be45564c868408e6514c8aabb83e211a0911 \
JACO_LITERT_ABI_SOURCE_MODE=download \
bash native/verify_litert_abi.sh
```

This downloads the four pinned upstream headers into a **temporary directory**,
compiles the native candidate with `-Werror=sign-compare`, and runs the
test callback. It does not access or replace any installed Android wheel,
nor execute actual Gemma decoding. Only two reviewed LiteRT SHAs may be used.
