#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
${CXX:-clang++} -O2 -std=c++17 -Wall -Wextra bench/native_probe.cpp -o bench/native_probe
echo 'Built bench/native_probe'
