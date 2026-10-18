# JACO Gemma INT8 × INT4 — native prototype

This is an independently executable ACFQ-family numerical kernel for a
**TFLite-style row-major [N,K] INT4 weight tensor** and INT8 activation vector.
It DOES NOT register as a LiteRT kernel or run inside Gemma's active decoder.

## Exact contract

- Packed signed, two's complement INT4 values in [-8,7]. Two elements
  per byte, **lowest-index element is in lower nibble**, contiguous [N,K].
- INT8 input x[K] with per-tensor `input_scale>0`, `input_zero_point`.
- Per-output-channel FP32 weight scales [N] and optional weight zero
  points [N], usually all zero but not assumed.
- Optional FP32 bias [N]. The output is **FP32**, computed as:
  `bias[j] + input_scale*weight_scale[j]*sum_i((x[i]-input_zp)*(w[j,i]-weight_zp[j]))`.
- `skip_exact_zero=1` skips only activations satisfying
  `x[i] == input_zero_point` and is mathematically lossless (not an
  approximate sparsity threshold). It is not assumed to be faster.
- Signed INT64 accumulation avoids overflow for large K.
- AArch64 NEON path operates on blocks of 32 in **dense** mode; scalar
  fallback handles sparse mode and odd-row pack alignment.
- Counters report only actual calls to this API. They **do not** imply
  the Gemma runtime is invoking this kernel.

## Build/test

```sh
cc -std=c11 -O3 -fPIC -shared native/gemma_int4_kernel.c -lm -o libjaco-gemma-int4.so
python3 -m unittest discover -s native -p 'test_*.py'
cc -std=c11 -O2 native/gemma_int4_kernel.c native/test_gemma_int4_kernel.c \
 -I native -lm -o /tmp/jaco-gemma-int4-check
/tmp/jaco-gemma-int4-check
```

GitHub Actions also cross-compiles the C smoke test to AArch64 and executes
it under qemu. The AArch64 NEON build must pass independently from x86 tests.

## Integration blockers

The installed Gemma 4 E2B `decode` graph contains 145 INT8 x INT4
FullyConnected operators, but the original LiteRT tensor buffers, exact
activation scale/zero-point metadata and backend dispatch ABI have not been
verified for this C contract. Its output is FP32, whereas LiteRT may
require an INT8 output with additional rounding/clamping rules and fused
activations. Do not attempt runtime injection or replace an installed LiteRT
library until a separate native integration is built and compared against
the untouched decoder on real token generation.

Sources: TFLite packed-int4 helpers in TensorFlow's portable_tensor_utils.h
and standard affine quantization semantics. Implemented semantics are
hypotheses for adapting those formats to our prototype, **not evidence
of actual Gemma kernel compatibility or speedup**.
