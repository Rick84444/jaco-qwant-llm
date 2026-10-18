# ACFQ native operator prototype

This is a **real, called C operator**, using the existing ACFQ GEMV kernel in
`acfq_kernel.c`, with explicit buffer lengths, supported formats, dimensions
and sparse index validation. Tests run the compiled implementation and compare
outputs with an independent scalar reference.

Build and run:

```sh
cc -std=c11 -O2 -fPIC -shared acfq_kernel.c native/acfq_backend.c -lm -o libjaco-acfq.so
python3 -m unittest discover -s native -p 'test_*.py'
```

Kernel contracts: weights are [K,N] **row-major by input channel**, N divisible
by 32, packed in the format described in `acfq_kernel.c`. For F32, the
input pointer must be correctly float-aligned. For quantized weights, caller
must supply ACFQ's own packed Q4/Q8 layout; these layouts have **not** been
established for installed Gemma 4 E2B. Indices are unique, sorted and within
[0,K). Null indices mean dense calculation (pass count=0). Passing an empty
non-null index buffer computes zeros.

**This is not a LiteRT operator registration, XNNPACK delegate, or replacement
for Gemma's native tensor operations.** It is the separately testable native
boundary required before integration. Production callers must also ensure
weights and outputs do not alias, allocated buffers remain alive, and
thread-safety/lifetime are managed. It does not patch the system LiteRT
library or claim any whole-model speed improvement.
