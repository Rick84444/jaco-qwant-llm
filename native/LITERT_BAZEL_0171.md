# Bazel-specific, pinned LiteRT v0.17.1 integration

GitHub Actions Android run `39477272375` passed Bazelisk, Android SDK,
NDK and ACFQ source modification, then failed with Bazel undeclared inclusions
for the injected `jaco_acfq_shadow.h`, native header, and native C source.

`prepare_litert_bazel_v0171.py` resolves exactly that issue **only** in
the isolated pinned LiteRT checkout already patched by
`patch_litert_acfq_v0171.py`. It adds a native C compilation unit and two
headers to the pinned `compiled_model` BUILD target. The C implementation is
removed from `compiled_model.cc`'s inline include to prevent double linkage.

Do not apply this helper to the Linux CMake prototype, an installed wheel,
other upstream revisions, or a running model. No performance or native
inference correctness is implied by a successful Bazel compile.

The helper requires the exact LiteRT revision, untouched original BUILD,
source backup matching Git HEAD, unique anchors, and a separate source tree.
`--check` is side-effect-free; `--apply` creates an original BUILD backup.

The Android workflow performs both operations after the source adapter.
Test locally with `python3 -m unittest discover -s native -p "test_prepare_litert_bazel_v0171.py"`
and review GitHub Actions native build evidence.
