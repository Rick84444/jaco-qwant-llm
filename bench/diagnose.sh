#!/bin/sh
# JACO: read-only Android llama-bench crash diagnostics.
# Does not start a second multi-GB model, kill processes or modify llama.cpp.
set -u
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
OUT="${1:-$HOME/jaco-llama-diagnostics.txt}"
case "$OUT" in -*) echo "Output path cannot start with -" >&2; exit 2;; esac
if [ ! -d "$LLAMA_DIR" ]; then
  echo "Missing llama.cpp directory: $LLAMA_DIR" >&2
  exit 2
fi
{
  echo "JACO Q1_0 crash diagnostics"
  date -u '+UTC %Y-%m-%dT%H:%M:%SZ'
  uname -a
  echo "=== llama source ==="
  git -C "$LLAMA_DIR" rev-parse --short HEAD 2>&1 || true
  git -C "$LLAMA_DIR" status --short 2>&1 || true
  echo "=== system tools ==="
  command -v clang++ || true
  command -v lldb || true
  command -v readelf || true
  echo "=== process state ==="
  pgrep -af 'llama-server|llama-bench' || true
  echo "=== memory ==="
  grep -E 'MemAvailable|SwapFree' /proc/meminfo || true
  echo "=== binaries ==="
  for bin in "$LLAMA_DIR/build-bench/bin/llama-bench" \
             "$LLAMA_DIR/build-safe/bin/llama-bench" \
             "$LLAMA_DIR/build/bin/llama-server" \
             "$LLAMA_DIR/build-vulkan/bin/llama-server"; do
    if [ -e "$bin" ]; then
      ls -lh "$bin"
      if command -v readelf >/dev/null 2>&1; then
        readelf -d "$bin" 2>/dev/null | grep NEEDED || true
      fi
    fi
  done
  echo "=== safe CMake configuration ==="
  if [ -f "$LLAMA_DIR/build-safe/CMakeCache.txt" ]; then
    grep -E '^(GGML_NATIVE|GGML_CPU_ARM_ARCH|GGML_VULKAN|CMAKE_BUILD_TYPE):' "$LLAMA_DIR/build-safe/CMakeCache.txt" || true
  fi
  echo "=== native CMake configuration ==="
  if [ -f "$LLAMA_DIR/build-bench/CMakeCache.txt" ]; then
    grep -E '^(GGML_NATIVE|GGML_CPU_ARM_ARCH|GGML_VULKAN|CMAKE_BUILD_TYPE):' "$LLAMA_DIR/build-bench/CMakeCache.txt" || true
  fi
  echo "=== note ==="
  echo "Exit 139 from both builds is not sufficient to attribute failure to Q1_0."
  echo "The LLDB crash was in thread linker64; symbols and mapped image remain unverified."
  echo "Do not execute the crashing model benchmark automatically."
} > "$OUT"
echo "Saved diagnostic evidence: $OUT"
echo "Next: from LLDB at SIGSEGV run 'image lookup --address <pc>' and 'image list -o -f'."
