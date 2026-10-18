# Real Android benchmark: experimental INT8 × INT4 kernel

Benchmarks the **separately compiled native candidate**, not Gemma token
generation. The matrices have observed Gemma decoder weight shapes:
`[256,1536]` and `[2048,1536]`. All values are synthetic.

Independent scalar reference: signed nibble decode and int64 accumulation,
followed by dequantization with per-tensor input scale, per-channel weight
scale and FP32 bias. This is compared to our dense AArch64 NEON kernel and
its exact-zero-skip option. Numerical checks and product counters must pass
before any result is printed.

Run on the Pixel 10 Pro in Termux:

```sh
cd ~/jaco-qwant-llm
git pull --ff-only origin main
cc -std=c11 -O3 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -I native \
  native/gemma_int4_kernel.c native/bench_gemma_int4.c \
  -lm -o gemma-int4-bench
./gemma-int4-bench --reps 9 | tee bench/gemma-int4-local.csv
```

One record is emitted per shape and synthetic exact-zero rate (0%, 25%,
50%). Times are median microseconds per GEMV, with 2 warm-up operations
per method. `dense_vs_scalar` and `exact_zero_vs_scalar` compare **only**
against this independent scalar reference, NOT Google's optimized LiteRT
backend. If speedup is below 1, the native candidate is slower than our
scalar reference. A sparse path may have no NEON usage: this is expected
and the speedup must be measured rather than assumed.

The output includes actual zero rate, numerical errors, number of dense
NEON blocks and skipped products. Copy the complete CSV output back to
the conversation for analysis. Measurements depend on thermal state,
CPU frequency, and memory; repeat when temperature is stable.
Never claim whole-model improvement from an operator microbenchmark.

A small synthetic `--quick --reps 2` case is run in CI as a smoke check.
