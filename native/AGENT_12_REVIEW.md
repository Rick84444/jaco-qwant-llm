# Verified review of local mission 12: delegate interception

## Disproved report references
The Gemma review reported line 1204 as `profiler->TriggerHook`, line
1205 as profiler error handling, and line 1211 as a profiler log. These
references are wrong for pinned upstream LiteRT
`44455548be35bde21737b8a28ef0d027632c282d`.

- Real delegate graph handoff: `compiled_model.cc` lines 1197–1199
  (`ModifyGraphWithDelegate`).
- Real graph counts: line 1204 (`GetGraphCounts`), 1205 (node delta).
- Real CPU delegation count: line 1211.
- Profiler hooks are processed earlier, around lines 1176–1194.

The patch registers a builtin TFLite FULLY_CONNECTED wrapper, not a
custom operator named `jaco_gemma_int4_gemv`. The profiler's
`CompilerStart` trigger is unrelated to proof of the ACFQ callback.

## Three distinct observations

1. `registration_installed=1`: resolver registration only.
2. `builtin_callback_invoked=1`: TFLite actually called the builtin
   wrapper. Bounded to the first eight invocations. Logs types and weight
   shape without printing weights or activations.
3. `native_invocations=1`: the separate ACFQ C kernel ran for eligible
   tensors, *without replacing the original inference output*.

Only phase (3) during a genuine Gemma decode would prove execution
inside model inference. None of these phases prove speed or numerical
parity with the original operator.

Tested by CI with pinned upstream TFLite tensor ABI, using a test
original-operator callback. A separately compiled upstream Linux
`liblitert_runtime.a` has been created by
https://github.com/Rick84444/jaco-qwant-llm/actions/runs/39298327347 .
The Android server on 9379 has not been modified or linked to that build.

If phase (1) occurs but (2) does not, inspect model op versions and
delegate partitioning. If phase (2) occurs but (3) does not, inspect
tensor types, dimensions and quantization gates.
