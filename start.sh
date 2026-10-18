#!/bin/sh
# One-shot start in Termux (or Linux): installs dependencies if missing,
# builds, self-tests the kernel and runs the synthetic benchmark.
# Usage: ./start.sh            synthetic benchmark
#        ./start.sh acts       benchmark on captured activations in ./acts
#        ./start.sh acts energy  same, plus battery energy per token
set -e
cd "$(dirname "$0")"
if [ -n "$PREFIX" ] && command -v pkg >/dev/null 2>&1; then
  for p in clang python python-numpy; do
    dpkg -s "$p" >/dev/null 2>&1 || pkg install -y "$p"
  done
fi
python3 -c "import numpy" 2>/dev/null || { echo "numpy missing: pip install numpy"; exit 1; }
./build.sh
stamp=$(date +%Y%m%d-%H%M%S)
if [ -n "$1" ]; then
  extra=""
  [ "$2" = "energy" ] && extra="--energy"
  python3 acfq_rig.py --data "$1" $extra --json "report-$stamp.json"
else
  python3 acfq_rig.py --synthetic --json "report-$stamp.json"
fi
