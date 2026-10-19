# Actual isolated LiteRT CPU build: first failure and correction

Pinned upstream LiteRT: `44455548be35bde21737b8a28ef0d027632c282d`
First build: https://github.com/Rick84444/jaco-qwant-llm/actions/runs/39262026494

**Observed:** CMake configuration SUCCEEDED; `compiled_model.cc.o`
(including the optional ACFQ shadow code) compiled successfully. At
`643/657`, the build failed in unmodified upstream
`litert/runtime/tensor_buffer.cc` at `#include "xnnpack.h"` with
`fatal error: xnnpack.h: No such file or directory`.

**Root cause, directly supported by source:** the initial CI job forced
`-DTFLITE_ENABLE_XNNPACK=OFF`, while upstream
`litert/runtime/tensor_buffer.cc` includes `xnnpack.h` whenever
`LITERT_DISABLE_CPU` is not defined. This is not an error from the
ACFQ native kernel itself.

**Correction:** run same isolated CPU build with
`-DTFLITE_ENABLE_XNNPACK=ON`, retaining GPU=OFF, NPU=OFF.
XNNPACK availability does NOT mean our builtin callback will execute;
delegate interception still needs explicit runtime tests.

**Verification still missing:** successful complete `litert_runtime`
static library, executable linking official LiteRT-LM, genuine Gemma
decoder invoking ACFQ, full operator parity, and speedup compared with
optimized LiteRT.

**Production isolation:** GitHub-hosted Linux runner only. The device's
Gemma HTTP server on port 9379 remains unchanged.
