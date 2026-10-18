# JACO Native C++ benchmark

A native probe of the **existing running** llama.cpp server. This is real
server-measured timing, NOT a new Q1_0 kernel or ACFQ integration.

Run on Termux:

    cd ~/jaco-qwant-llm
    sh bench/build.sh
    ./bench/native_probe --port 11435 --reps 2 --predict 4

CSV reports prompt and generated token throughput as measured by llama.cpp.
Temperature, threading, CPU governor and prompt caching affect results.
Avoid comparing runs with different model/context/thread settings.

To compare CPU and Vulkan: stop the existing llama-server first to avoid running
two copies of the 3.6GB model in memory. Run the benchmark against an otherwise
identical CPU server; then replace that server with your existing
~/llama.cpp/build-vulkan/bin/llama-server, using conservative partial offload
(-ngl 8) and repeat. Check /health before running the benchmark. Vulkan offload
is unverified on this handset; do not assume speedup.

This tool never stops a process, edits model files, or pretends the Q1_0
dequantization kernel has been optimized.
