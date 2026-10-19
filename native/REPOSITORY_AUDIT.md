# Repository audit — reproducible technical debt control

This project is an **experimental component**, not a fully integrated
LiteRT-LM kernel. Nothing in this checklist represents sixteen independent
assistants already executed or external code modifications completed by them.

## Quality model

Sixteen **specialist review assignments** live in
`native/repo_audit_missions.json`. Each is its own prompt/context and
documents the source paths, expected output, uncertainties and falsifying
experiment. If the local model is running, explicitly execute:

```bash
cd ~/jaco-qwant-llm
python3 native/run_litert_agents.py --audit-repo --dry-run
python3 native/run_litert_agents.py --audit-repo --run
```

If any existing report was produced from an earlier prompt/source revision,
its status is `stale_existing`, the file is retained without modification,
and the runner exits with status 2 rather than claiming a current review.
Exit status 1 means a model/source error; 0 means current reviews.

Reports are saved **outside the checkout** under
`~/jaco-repo-audit-reports`. No task has authority to write GitHub,
edit code, execute suggested shell commands, or approve itself.
These are sequential local model reviews, not 16 concurrent daemons.
Do not treat their recommendations as verified until tests reproduce them.

## Issues observed and addressed in code (2026-10-19)

| Scope | Reproducible evidence | Mitigation | Remaining risk |
|---|---|---|---|
| Review evidence | Old reports were silently skipped after source changes; unnumbered excerpts allowed inaccurate line references | Full-file fingerprints, bounded numbered excerpts, explicit `stale_existing` classification | Reports generated with old prompts are still historical, not fresh reviews |
| Data durability | Generated Markdown/manifest were directly overwritten | Same-filesystem temp + atomic replace; sha-based backups on explicit force; reject symlink destinations | Power-loss durability varies with filesystem; no persistent cross-file transaction |
| Review endpoint | Arbitrary `--base-url` could send local source extracts to remote endpoints | Require explicit localhost HTTP port | Browser and native process trust remain separate; local processes can contact localhost |
| Local adapter | Browser-simple cross-origin POST could consume inference resources | Require JSON, check Origin and loopback Host (DNS rebinding defense) | No multi-tenant authentication; do not bind to a public interface |
| Pinned LiteRT build | Official Linux `liblitert_runtime.a` compiled, but not linked into Android Gemma | Keep production server unchanged | True Gemma FullyConnected callback not yet observed |
| End-to-end acceleration | Synthetic NEON kernel beat *our scalar reference*, not optimized LiteRT | Explicitly reject whole-model speed claims | Model packing, real weight data, quantization and token/s parity pending |

## Highest-priority unresolved gates

1. Identify the native LiteRT revision actually loaded by Termux LiteRT-LM.
2. Verify source-compatible `FULLY_CONNECTED` operator registration on that revision; avoid borrowing source anchors from a different version.
3. Create a separately linked **executable**, not merely a static library.
4. Trace actual Gemma 4 E2B decoding from registration → delegation →
   callback → successful ACFQ invocation.
5. Verify correct weights, per-channel scaling, bias, output requantization,
   identical content with original executor, thermal behavior and tokens/s.
6. If the model graph delegates all relevant nodes elsewhere or numerical
   fidelity cannot match the original optimized backend, retain stock Gemma.

Do not delete research files merely to make the repo appear smaller.
Remove an artifact only after verifying it is unused, preserving provenance
and passing checks. Changes should be small, test-backed commits to main.
