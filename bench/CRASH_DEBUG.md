# Debugging Bonsai Q1_0 llama-bench crash on Android

The user observed `llama-bench` SIGSEGV (139) in both `build-bench`
(native ARM optimizations) and `build-safe` (armv8-a). A separate
`llama-server` successfully loaded and generated tokens from the same GGUF.
At LLDB the crashing thread was `linker64`, `pc=0x7ff02f3980`,
`x0=0`, fault address `0x104`, but no module name or disassembly
was obtained. **This does not establish a Q1_0 kernel defect**.
Android dynamic loader, library dependencies, runtime initialization,
mismatched binaries, benchmark code, and model loading remain hypotheses.

## Safe first step

Run only this read-only diagnostic from Termux:

```sh
cd ~/jaco-qwant-llm
sh bench/diagnose.sh
cat ~/jaco-llama-diagnostics.txt
```

This does not allocate another copy of the 3.6 GB model or kill services.

## Obtain a meaningful stack

Use LLDB on the locally compiled benchmark and reproduce the fault
**only after stopping other memory-heavy model processes**:

```sh
lldb -- ~/llama.cpp/build-safe/bin/llama-bench \
  -m "$HOME/models/Bonsai-27B-Q1_0.gguf" -p 4 -n 1 -t 2 -ngl 0 -r 1
```

At `(lldb)`, execute `run`, followed by:

```
thread backtrace all
image lookup --address 0x0000007ff02f3980
image list -o -f
disassemble --pc --count 12
register read pc lr x0 x1
```

The address above is from an earlier run and *can change* because of
ASLR. In a new LLDB session, use the current program counter
(`register read pc`) for the `image lookup`.

If no symbols resolve, try a debug build in a **new** directory using
`-DCMAKE_BUILD_TYPE=Debug` and `-DGGML_NATIVE=OFF`, leaving known-good
server binaries untouched. Do not patch GGML or claim speedup before the
crash site is identified.

This issue belongs to the user's separate local `~/llama.cpp` checkout.
GitHub changes in this repository cannot fix that executable until the
actual fault is established and patched in the relevant source.
