# Refresh an existing sparse LiteRT checkout on Termux

The original `setup_litert_acfq_source.sh` intentionally refuses to
overwrite an existing checkout. Your existing
`~/jaco-litert-acfq-source` is still pinned to upstream commit
`44455548be35bde21737b8a28ef0d027632c282d`, but retains the
old JACO hook from the first preparation.

Run from `~/jaco-qwant-llm` after pulling main:

```bash
python3 native/refresh_litert_acfq_source.py --check
python3 native/refresh_litert_acfq_source.py --apply
grep -n JACO_ACFQ_DELEGATE_TRACE_V1 ~/jaco-litert-acfq-source/litert/runtime/compiled_model.cc
```

The helper verifies the exact upstream Git revision and preserved
`compiled_model.cc.jaco-original`; it only accepts precisely generated
earlier or current patch text, and known shadow-header revisions from the
JACO repository. It creates separate pre-refresh backups and refuses to
overwrite unexpected manual work.

The resulting source adds callback and aggregate delegation diagnostics,
but **does not build, install, or replace LiteRT-LM**. The Gemma 4 E2B
service on localhost:9379 is unchanged, so its healthy `/v1/models`
endpoint cannot demonstrate execution of this patch. Genuine inference
integration requires a separately linked LiteRT-LM test binary.
