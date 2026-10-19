#!/usr/bin/env bash
# ABI validation with authentic pinned LiteRT headers, without linking LiteRT.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="44455548be35bde21737b8a28ef0d027632c282d"
SOURCE="${JACO_LITERT_SHADOW_SRC_DIR:-$HOME/jaco-litert-acfq-source}"
MODE="${JACO_LITERT_ABI_SOURCE_MODE:-git}"
BASE="${TMPDIR:-${PREFIX:+$PREFIX/tmp}}"
BASE="${BASE:-/tmp}"
CC="${CC:-cc}"
CXX="${CXX:-c++}"
mkdir -p "$BASE"
WORK="$(mktemp -d "$BASE/jaco-litert-abi.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
if [[ "$MODE" == "git" ]]; then
  [[ -d "$SOURCE/.git" ]] || { echo "Isolated checkout absent: $SOURCE" >&2; exit 2; }
  [[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$PIN" ]] ||
    { echo "Wrong upstream revision; refusing ABI test" >&2; exit 2; }
elif [[ "$MODE" != "download" ]]; then
  echo "Invalid header source mode" >&2; exit 2
fi
for f in \
  tflite/core/c/common.h \
  tflite/core/c/c_api_types.h \
  tflite/converter/core/c/tflite_types.h \
  tflite/core/kernels/builtin_op_kernels.h; do
  mkdir -p "$WORK/$(dirname "$f")"
  if [[ "$MODE" == "git" ]]; then
    git -C "$SOURCE" show "$PIN:$f" > "$WORK/$f"
  else
    curl --fail --silent --show-error --location --retry 2 \
      "https://raw.githubusercontent.com/google-ai-edge/LiteRT/$PIN/$f" \
      -o "$WORK/$f"
  fi
  test -s "$WORK/$f"
done
mkdir -p "$WORK/litert/runtime/jaco_acfq_native"
cp "$REPO/native/litert_acfq_shadow.h" "$WORK/litert/runtime/jaco_acfq_shadow.h"
cp "$REPO/native/gemma_int4_kernel.h" \
  "$WORK/litert/runtime/jaco_acfq_native/gemma_int4_kernel.h"
"$CC" -std=c11 -O2 -Wall -Wextra -I "$REPO/native" \
  -c "$REPO/native/gemma_int4_kernel.c" -o "$WORK/gemma_int4.o"
"$CXX" -std=c++17 -O2 -Wall -Wextra -I "$WORK" \
  "$REPO/native/test_litert_acfq_real_abi.cc" "$WORK/gemma_int4.o" \
  -lm -o "$WORK/real-abi"
"$WORK/real-abi" > "$WORK/results.log" 2>&1
cat "$WORK/results.log"
grep -q 'JACO_ACFQ_SHADOW: native_invocations=1' "$WORK/results.log"
grep -q 'JACO_ACFQ_SHADOW: builtin_callback_invoked=1 invocation=1' "$WORK/results.log"
grep -q 'JACO_ACFQ_SHADOW: builtin_callback_invoked=1 invocation=2' "$WORK/results.log"
grep -q 'JACO real-header ABI PASS' "$WORK/results.log"
echo 'PASS: authentic upstream TFLite ABI; no runtime/Gemma integration claimed'
