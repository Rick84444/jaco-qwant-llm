# Agent 13 — GPU route: verified review

**Verdict:** Plausible bypass hypothesis, **no demonstrated Gemma-specific route**.

LiteRT revision `44455548be35bde21737b8a28ef0d027632c282d`:
- `compiled_model.cc:1122`: loop through registered accelerators.
- `compiled_model.cc:1135-1138`: JIT-delegate eligibility check. These
  lines do not implement GPU-over-CPU priority.
- `compiled_model.cc:1197-1199`: call to `ModifyGraphWithDelegate`.
- `compiled_model.cc:1204-1220`: `GetGraphCounts`, per-accelerator
  delegated-node and partition counts.
- `compiled_model.cc:1233-1245`: full delegation flag and CPU fallback check.

The agent's suggestion to reroute more operations to GPU is rejected:
ACFQ is an ARM64 CPU kernel. Changing GPU routing risks regressions and does
not establish that the INT8×INT4 FullyConnected node can reach ACFQ.

**Implemented diagnostic:** an opt-in aggregate per-accelerator delegated
node log in the source patcher. It does not modify delegation or model data.
The true discriminant remains a real Gemma decode on a separately rebuilt
LiteRT-LM linked with patched LiteRT:

1. `registration_installed=1` — registration only.
2. `delegate_applied=...` — aggregate graph coverage only.
3. `builtin_callback_invoked=1` — CPU wrapper actually invoked.
4. `native_invocations=1` — ACFQ kernel executed, original result retained.

The current modified Linux `liblitert_runtime.a` build succeeded in Actions
but it is **not** connected to the running Android LiteRT-LM Python service.
