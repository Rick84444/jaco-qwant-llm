# Native LiteRT FullyConnected source hook — experimental shadow integration

**Status:** implementation committed for a manually built isolated LiteRT
source checkout; **not built into the running Gemma service**. The hook
registers a native TFLite Builtin `FULLY_CONNECTED` callback that calls
Google's original operator first, then (for a narrowly matching INT8 × INT4
single-vector shape) calls `jaco_gemma_int4_gemv` with the actual tensor
activation buffer, packed weights and channelwise quantization parameters.

The callback preserves Google's output completely. It logs
`JACO_ACFQ_SHADOW: native_invocations=1` **only when the native kernel
actually executes**. Registration alone logs `native_invocations=0`.
There is a process-wide limit of one eligible shadow invocation to bound
overhead, and no claim about output numerical equivalence or acceleration.

## Why a source build is necessary

The installed LiteRT-LM wheel exposes no verified hook to replace built-in
TFLite `FULLY_CONNECTED` with an arbitrary dynamically loaded kernel. The
compiled model uses a TFLite builtin resolver and then applies accelerator
delegates; XNNPACK/YNNPACK or GPU may claim a supported node before our CPU
wrapper can execute. In those cases **native invocation count stays zero**;
a registration message is not enough. No attempt is made to disable
delegation or overwrite production libraries. The documented alternative
LiteRT CustomOpDispatcher applies to custom operators, not automatically to
built-in FULLY_CONNECTED in an existing model.

Source references (as inspected on 2026-10-19):
- `google-ai-edge/LiteRT`: `litert/runtime/compiled_model.cc` creation
  of `BuiltinOpResolverWithoutDefaultDelegates`, then
  `InterpreterBuilder` and `ModifyGraphWithDelegate`.
- `google-ai-edge/LiteRT`: `tflite/core/kernels/register.cc`, 
  `BuiltinOperator_FULLY_CONNECTED` registered for versions 1–15.
- `google-ai-edge/LiteRT-LM`: 
  `runtime/executor/llm_litert_compiled_model_executor.cc`, which calls
  `CompiledModel::Create`; LiteRT-LM CPU setup uses XNNPACK flags.

## Usage in a separate source checkout only

```bash
# Only on an isolated, separately fetched LiteRT source tree:
python3 native/patch_litert_acfq.py --litert-root ~/src/LiteRT --check
python3 native/patch_litert_acfq.py --litert-root ~/src/LiteRT --apply
```

This adds our native C kernel and header to the checkout and modifies its
`litert/runtime/compiled_model.cc`. The original C++ source is backed up as
`compiled_model.cc.jaco-original`. Source changes are opt-in and off by
default: set `JACO_ACFQ_SHADOW=1` for an **isolated rebuilt** CPU LiteRT
test binary to register the shadow callback. Build instructions depend on
the exact LiteRT/ LiteRT-LM revision and link configuration; **there is no
claimed working rebuilt binary or live Gemma run as of this commit**.

Prerequisites to claim integration success:
1. Compile this source against an exact LiteRT revision and run synthetic
   TFLite model test first; ensure original output remains unchanged.
2. Run actual Gemma 4 E2B through isolated LiteRT-LM linked to that build.
3. Verify log `native_invocations=1` in a real decode request and inspect
   eligibility and accelerator delegation if it stays 0.
4. Separate steps: verify output requantization and bias, implement
   correct **replacement** instead of shadow, compare responses and
   throughput with the untouched LiteRT baseline.

Only `registration_installed=1` is insufficient. The live model on
port 9379 is not affected by this source patch.

## C++ bridge verification

A separate GitHub Actions smoke test compiles `litert_acfq_shadow.h` against
minimal TFLite-like **test structs**, links the actual C kernel, executes a
simulated `FULLY_CONNECTED` callback, and asserts the original output was
unchanged and the ACFQ invocation was logged.

This does **not** constitute upstream LiteRT ABI compilation or a Gemma
integration proof. An isolated build against the real LiteRT headers and
then real Gemma runtime execution are still required.

The wrapper imports `tflite/core/kernels/builtin_op_kernels.h` for
`Register_FULLY_CONNECTED()` (the generic resolver header alone does not
declare individual builtin registration functions).
