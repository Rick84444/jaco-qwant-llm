# Final LiteRT integration: sixteen scoped technical missions

**IMPORTANT:** These are independently testable assignments; they do not
represent sixteen autonomous agents running in this chat. Do not mark a
mission done without an attached command log, diff or numerical evidence.
Only edit the isolated LiteRT checkout. Stock Gemma on port 9379 remains
the production reference.

| ID | Specialist mission | Acceptance criterion | State |
|---|---|---|---|
| 01 | Provenance | Frozen official LiteRT SHA and isolated checkout | Verified |
| 02 | ABI hygiene | Remove unnecessary register.h dependency | Patched |
| 03 | Real header ABI | Compile real TFLite structs and call native kernel | Automated check added; verify CI |
| 04 | NEON safety | Pixel AArch64 kernel agrees with scalar reference | Previously verified in local/CI tests |
| 05 | INT4 packing | Capture actual row stride and nibble order | Pending |
| 06 | Quant metadata | Compare real per-axis scales and zero points | Pending |
| 07 | Bias | Verify complete FullyConnected output including bias | Pending |
| 08 | Requantization | INT8 output parity and saturation/rounding | Pending |
| 09 | Operator versioning | Match actual Gemma FC operator version range | Pending |
| 10 | Tensor shapes | Validate decoder-step input/output layout | Pending |
| 11 | CPU-only path | Run separate genuine LiteRT CPU interpreter | Pending |
| 12 | XNNPACK/YNNPACK | Prove op not bypassed by delegate | Pending |
| 13 | GPU backend | Verify production GPU path and interception possibility | Pending |
| 14 | Android build | Link isolated rebuilt LiteRT into separate test runtime | Pending |
| 15 | Real decode | Model prints native invocation during actual token generation | Pending |
| 16 | A/B and rollback | Same-backend parity, latency; never replace working 9379 | Pending |

## Evidence ladder

1. Native C and mock C++ callback: already observed on Pixel.
2. Genuine pinned TFLite tensor ABI test: use
   CC=clang CXX=clang++ bash native/verify_litert_abi.sh
   after fetching this repo into Termux. Header blobs come from the
   existing pinned partial Git checkout. Script does not modify the
   checkout or running LiteRT-LM installation.
3. Real LiteRT FullyConnected interpreter and actual Gemma token decode:
   **not demonstrated**.
4. Replacing original operator and measurable speed gain:
   **not demonstrated**.

**Stop rule:** If actual Gemma computation is GPU-delegated, our CPU
builtin hook may never fire. If replacement cannot match reference
numerics or cannot outperform optimized LiteRT, stop and retain stock
LiteRT-LM. A shadow callback proves invocation only, never acceleration.
