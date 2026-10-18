#!/bin/sh
# JACO component runtime: no automatic installs and no synthetic fake responses.
set -eu
cd "$(dirname "$0")"
if ! command -v node >/dev/null 2>&1; then
  echo "Node.js required. On Termux: pkg install nodejs" >&2
  exit 1
fi
exec node runtime/cli.mjs "$@"
