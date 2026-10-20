# JACO model evaluation – 20 October 2026

## Scope and confidence

Verified **model cards and architecture/licensing** (high confidence in what
authors publish) are separated from **vendor benchmarks** and **community
experience** (hypotheses, not JACO experiment results). No real JACO
multi-step tool benchmark or paid HF inference call has been performed here.

## Shortlist and practical deployment

| Model | JACO opportunity | Deployment fit | Evidence |
| --- | --- | --- | --- |
| Qwen3-Coder-Next | Code agents/tool chains, 80B total/3B active | Workstation with memory or HF provider | https://huggingface.co/Qwen/Qwen3-Coder-Next |
| DeepSeek-V4.1-Flash | Long-context code agents, async prefill/decode architecture | Hosted; not phone | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash |
| Qwen3.8-27B | Coding, vision, quantized local experiments | High-memory workstation/hosted; phone not verified | https://huggingface.co/Qwen/Qwen3.8-27B |
| Kimi K3 | Very long coding/knowledge work | Hosted; custom Kimi K3 license | https://huggingface.co/moonshotai/Kimi-K3 |
| GLM-5.3-Flash | Coding and agent planning | Hosted; 320B total even though 18B active | https://huggingface.co/zai-org/GLM-5.3-Flash |
| Gemma 4 E2B | Android fallback/offline router | Existing LiteRT-LM baseline; ACFQ NOT integrated | https://huggingface.co/google/gemma-4-E2B |
| gpt-oss-20b | Reasoning/tool-use control group | Workstation/hosted; must honor Harmony format | https://huggingface.co/openai/gpt-oss-20b |

**Do not confuse MoE active parameter count with weight storage.**
**Do not assume a model on Hugging Face Hub is served via the router.**

## Independent reviews / tension

- Developer experience: https://www.reddit.com/r/LocalLLaMA/comments/1sxqa2c/im_done_with_using_local_llms_for_coding/
  Criticism of 27B decision-making and tool calls compared with cloud systems.
- Developer experience: https://www.reddit.com/r/LocalLLM/comments/1wr3sut/best_coding_agent_llm/
  Qwen3-Coder-Next regarded favorably in a small practical coding comparison.
- Agent tool chains: https://www.reddit.com/r/better_claw/comments/1uf9ogl/tested_qwen_gemma_glm_and_minimax_for_agent/
  Multi-hop reliability matters more than success on a single example.

User reports are anecdotal and task-dependent. Do not assign 10/10 performance
scores from these sources.

## Hugging Face live signals

Official API: https://huggingface.co/docs/inference-providers/hub-api
- `GET https://router.huggingface.co/v1/models` exposes provider entries,
  optionally `status`, `supports_tools`, `supports_structured_output`,
  per-1M-token prices, context length and throughput.
- Provider attributes may be absent. Missing `supports_tools` is NOT true.
- Providers/models may change. Poll only when the user selects a new model.
- A public demo must not contain operator secrets or make unrestricted paid
  requests; this Space only fetches public metadata and links to model cards.
- A local Gemma model remains in place when HF is unavailable or disallowed.

## Reproducible next-stage JACO acceptance test (planned, NOT run)

1. Fixed dataset of >=30 tasks across: repo change + test, tool JSON fidelity,
   three-to-ten tool calls, branch diff review, regression rollback,
   long-context recall and refusal of fabricated tool success.
2. Same JACO supervisor, prompts and tool set across models. Record exact
   model revision, inference provider and quantization; seed where supported.
3. Output: task success, executable patch validity, tests passed,
   tool-schema error rate, time-to-first-token, tokens/s,
   cost/job, retries, operator intervention and failure recovery.
4. Score successful **end-to-end** jobs, not single-message quality.
   Mark unknown provider capability as NOT VERIFIED.
5. No autonomous charge-incurring routing; human chooses HF provider/model,
   sets per-job max tokens and authorizes any external writes/payments.
6. Benchmark on Pixel 10 Pro separately, measure sustained heat/memory.
   Do not start another massive local model beside Gemma LiteRT.
7. Store results as evidence and let future JACO selections learn from
   **verified runs**, not promotional benchmark tables.

## Product implementation

`runtime/model-intelligence.mjs`: local catalog and deterministic rankings.
`runtime/provider.mjs`: retains provider metadata, not just model IDs.
`./start.sh recommend agents`: offline hypothesis shortlist.
`./start.sh rank agents`: live router metadata; excludes providers without
explicitly supported tool calls.
`demo/huggingface-space/catalog.json`: public curated model list.
`demo/huggingface-space/index.html`: goal tabs and per-model live metadata
check (public, optional, fails closed on CORS/network restrictions).

## 20 October late-update candidates

- **DeepSeek V4.1-Flash-1017 (17 October release)**:
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
  Official model card reports improvements over V4.1 Flash on agentic benchmarks
  and runtime. Do not equate that self-report with JACO validation. Inference
  availability changed between cached model-card snapshots; rely on the *live*
  router's status at selection time.
- **Qwen3.8-Flash-Next**:
  https://huggingface.co/Qwen/Qwen3.8-Flash-Next
  180B-class model; Qwen community license and workload efficiency merit
  caution before choosing for a public demo.
- October community example:
  https://www.reddit.com/r/AI_Agents/comments/1x94evs/chinese_models_for_vibe_coding_kimi_deepseek_glm/
  Developers use Kimi for architecture/design, GLM Flash for execution, and
  independent review for critical changes. This is an unvalidated **agent
  role-specialization hypothesis**, not a proven multi-agent speedup.
- October multi-model comparison:
  https://www.reddit.com/r/LocalLLaMA/comments/1x8bb6b/deepseek_v41_flash_1017_now_leads_openweight/
  Discussion highlights disagreement about benchmark rank versus useful output
  per dollar/token. JACO should record task success plus tokens and elapsed time.
