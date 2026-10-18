#!/bin/sh
# Builds libacf_rig.so (fp32 rig) and libacfq.so (quantized rig), then runs
# the kernel self-test. Termux aarch64 or Linux x86-64.
set -e
cd "$(dirname "$0")"
CC=${CC:-clang}
command -v "$CC" >/dev/null || CC=gcc
build() {
  $CC -O3 -march=native "$@" 2>/dev/null || $CC -O3 -mcpu=native "$@" 2>/dev/null || $CC -O3 "$@"
}
build -shared -fPIC acf_kernel.c  -o libacf_rig.so -lm
build -shared -fPIC acfq_kernel.c -o libacfq.so    -lm
build test_acfq.c -o test_acfq -lm
./test_acfq
echo "built: libacf_rig.so libacfq.so"
