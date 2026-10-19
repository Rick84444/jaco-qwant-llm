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

## Observed Pixel 10 Pro / Termux run — 2026-10-19

**Provenance:** terminal log supplied by the user, using standalone
`native/bench_gemma_int4.c` at repo revision `7c1d19f`, compiled with
`cc -std=c11 -O3 -Wall -Wextra -D_POSIX_C_SOURCE=200809L`; `--reps 9`.
This is not an independent rerun on a developer-controlled Pixel. CPU
frequency/temperature were not recorded and no raw per-iteration latencies
are available. Each reported latency is the program's median of 9
measurements, with two warm-ups; methods execute in fixed order (scalar,
dense, exact-zero). Synthetic, deterministic values have observed matrix
dimensions, not the actual installed model's weights.

Raw user-reported output: [`evidence/gemma-int4-pixel10pro-2026-10-19.csv`](../evidence/gemma-int4-pixel10pro-2026-10-19.csv).

| Gemma-shaped weights | Target zero fraction | Scalar, µs | Dense NEON, µs | Exact-zero, µs | Dense / scalar |
| --- | ---: | ---: | ---: | ---: | ---: |
| [256,1536] | 0% | 971.953 | 109.375 | 795.781 | 8.8864× |
| [256,1536] | 25% | 666.771 | 80.963 | 524.141 | 8.2355× |
| [256,1536] | 50% | 587.187 | 74.948 | 409.244 | 7.8346× |
| [2048,1536] | 0% | 2608.907 | 311.068 | 2358.438 | 8.3869× |
| [2048,1536] | 25% | 2353.282 | 278.568 | 2024.609 | 8.4478× |
| [2048,1536] | 50% | 2345.104 | 284.401 | 1689.922 | 8.2458× |

All six reported numerical absolute-error metrics printed as
`0.00000000` to eight decimal places. That is **not** proof of bitwise
identity; independent scalar-reference tolerance checks passed.
`dense_neon_blocks` was 12,288 for N=256 and 98,304 for N=2048.
At a *target* zero fraction of 0%, the actual zero fraction was 0.26%,
because the synthetic input generator can produce the zero point naturally.

**Evidence-backed conclusion:** the dense NEON implementation beat
**our scalar reference** in every observed case by **7.8346–8.8864×**.
The scalar exact-zero-skip path was slower than dense NEON in all six
cases, including 50% zeros. Do not prioritize a standalone scalar sparse
path for Gemma performance without a new, better implementation.
This is **not** a claim about gains against Google's LiteRT or token/s;
our kernel is still not connected to the running Gemma model.

To check the committed CSV's headers, numeric columns, derived speed
ratios, skipped-product counters and expected six conditions:

```sh
python3 native/verify_gemma_int4_benchmark.py
python3 -m unittest discover -s native -p 'test_*.py'
```

**Next engineering gate:** inspect actual packed weights and per-tensor
activation/output quantization and compare an isolated, supported native
operator integration against the original LiteRT execution path using
identical inputs. Do not replace working runtime binaries prematurely.
