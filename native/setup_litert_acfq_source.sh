#!/usr/bin/env bash
# Isolated, pinned LiteRT source preparation for the JACO native shadow hook.
# This deliberately does NOT compile or replace the user's running LiteRT-LM.
set -euo pipefail

JACO_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="https://github.com/google-ai-edge/LiteRT.git"
PIN="44455548be35bde21737b8a28ef0d027632c282d"
DEST="${JACO_LITERT_SHADOW_SRC_DIR:-$HOME/jaco-litert-acfq-source}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
command -v git >/dev/null || die "git is required"
command -v python3 >/dev/null || die "python3 is required"
[[ -f "$JACO_REPO/native/patch_litert_acfq.py" ]] || die "run from a complete JACO checkout"
[[ "$DEST" == "$HOME/"* ]] || die "destination must be under HOME"
[[ "$DEST" != "$HOME/.litert-lm"* && "$DEST" != "$HOME/jaco-qwant-llm"* ]] ||
  die "refusing to touch model files or JACO checkout"
[[ ! -e "$DEST" ]] || die "destination already exists: $DEST (no overwrite)"

available_kb="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
[[ "$available_kb" =~ ^[0-9]+$ ]] || die "cannot read free disk space"
(( available_kb >= 2097152 )) ||
  die "less than 2 GiB free; refusing even a sparse source checkout"

printf 'Preparing isolated LiteRT source checkout:\n  %s\n' "$DEST"
printf 'Pinned upstream revision: %s\n' "$PIN"
printf 'Your LiteRT-LM model and server on port 9379 are untouched.\n'

# Git partial clone avoids downloading the full upstream source tree on Android.
git clone --quiet --filter=blob:none --depth=1 --no-checkout "$UPSTREAM" "$DEST"
git -C "$DEST" sparse-checkout set --no-cone \
  '/litert/runtime/compiled_model.cc' '/tflite/core/c/common.h'
# Resolve the exact upstream commit instead of depending on moving main.
git -C "$DEST" fetch --quiet --depth=1 origin "$PIN"
git -C "$DEST" checkout --quiet --detach "$PIN"

python3 "$JACO_REPO/native/patch_litert_acfq.py" \
  --litert-root "$DEST" --check
python3 "$JACO_REPO/native/patch_litert_acfq.py" \
  --litert-root "$DEST" --apply
test -s "$DEST/litert/runtime/compiled_model.cc.jaco-original"
test -s "$DEST/litert/runtime/jaco_acfq_shadow.h"
test -s "$DEST/litert/runtime/jaco_acfq_native/gemma_int4_kernel.c"

printf '\nPASS: source hook prepared in %s\n' "$DEST"
printf 'STATUS: source patch only; no native library or Gemma runtime integration built.\n'
printf 'Do NOT expect native_invocations=1 in the running model yet.\n'
