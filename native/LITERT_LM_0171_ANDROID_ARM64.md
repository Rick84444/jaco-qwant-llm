# Complete LiteRT-LM v0.17.1 Android ARM64 native build

This is an **experimental full LiteRT-LM native shared-library build**, not
merely the LiteRT static runtime. It does not replace the installed Gemma
or Termux wheel, and it does not produce a verified acceleration claim.

Source pins: LiteRT-LM 5e58e9a0aef7abf7091207a8b1d1063a1c800f08;
LiteRT 9fe5be45564c868408e6514c8aabb83e211a0911;
Bazel 7.6.1; Android NDK r28b; architecture arm64-v8a.

Upstream target //c:litert-lm is a cc_binary with linkshared=1.
Upstream Python wheel can use prebuilt libraries, so building a wheel alone
would not prove the JACO ACFQ source was included.

Workflow .github/workflows/litert-lm-0171-android-arm64.yml uses Bazel
--override_repository=litert=... to compile source with JACO registration
in the pinned dependency. It validates ELF AArch64/DYN, Android libc
and libandroid dependencies, and the hook's string marker.

Output artifact: experimental-litert-lm-0171-android-arm64.

Never overwrite site-packages/litert_lm/liblitert-lm.so with this artifact.
Before production: separate on-device loading, real Gemma decode callback,
ACFQ native invocation, correctness incl. bias/quantization and reliable
tokens/s speedup must all be independently verified.
Current shadow calls the original kernel and therefore is NOT an accelerator
replacement, even when compiled successfully.
