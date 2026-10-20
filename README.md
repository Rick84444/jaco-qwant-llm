# JACO Model Runtime

A portable model-runtime component for JACO. It speaks OpenAI-compatible chat to a REAL configured model provider and retains ACFQ as an experimental ARM/NEON GEMV kernel. Later, JACO-CODE can import the provider adapter or call the local API. No duplicate agent, fake chat gateway or competing JACO supervisor is introduced.

**Current limitations:** ACFQ is NOT connected to any language model's inference graph. No token/s, model perplexity, or energy improvement has been proven by this adapter. Real model replies are forwarded from a working upstream service.

## Termux / Linux quickstart

Node.js 20 or newer is required. On Termux:

    pkg install nodejs
    git clone https://github.com/Rick84444/jaco-qwant-llm.git
    cd jaco-qwant-llm

Start your actual model server separately (e.g. llama.cpp with its OpenAI-compatible API).

    export JACO_MODEL_BASE_URL=http://127.0.0.1:11435/v1
    # Set JACO_MODEL_ID if multiple models are listed by the provider.
    ./start.sh doctor
    ./start.sh chat "Hej JACO"
    ./start.sh serve

The default runtime listens ONLY on localhost port 3131. Requests to `/v1/chat/completions` require `Content-Type: application/json` and reject cross-origin browser origins; the adapter provides no CORS access for unrelated websites. Ctrl+C stops it. No automatic installs and no unprotected port 8000 server.

## Gemma 4 E2B on Android / LiteRT-LM

Use an already installed LiteRT-LM model instead of building llama.cpp or
loading a second copy of Bonsai. Run `litert-lm serve --host 127.0.0.1
--port 9379` in a separate terminal only if the provider is not running.
Check `curl -fsS http://127.0.0.1:9379/v1/models`: it must list
`gemma4-e2b`.

Configure the **same** provider-neutral JACO Model Runtime:

```sh
cd ~/jaco-qwant-llm
export JACO_MODEL_BASE_URL=http://127.0.0.1:9379/v1
export JACO_MODEL_ID=gemma4-e2b
export JACO_MODEL_THINKING=provider
./start.sh doctor
./start.sh chat "Svara med ett kort hej"
./start.sh serve
```

Run `./start.sh serve` in its own Termux session. In another session:

```sh
curl -fsS http://127.0.0.1:3131/v1/models
curl -fsS http://127.0.0.1:3131/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hej"}],"max_tokens":32}'
```

The upstream LiteRT endpoint remains on **9379**, adapter on **3131**,
and the existing JACO Core owns **3000**. Do not start a second
llama-server or bind the model directly to 3000. `doctor` checks
discovery only; the `chat` request verifies actual inference.
No JACO-CODE agent tools or memory are automatically connected by
this component. Configure those in the JACO Core integration.

## API for future JACO-CODE integration

| Path | Contract |
| --- | --- |
| GET /health | 200 if an actual upstream model is discoverable and selectable; otherwise 503. Model discovery is NOT inference proof. |
| GET /capabilities | Kernel availability and honest ACFQ inference status (currently disabled). |
| GET /v1/models | Model IDs returned by the upstream API. |
| POST /v1/chat/completions | Forwards the request to the real provider and returns its response, including SSE streaming. |

Try:

    curl http://127.0.0.1:3131/health
    curl http://127.0.0.1:3131/capabilities
    curl http://127.0.0.1:3131/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"messages":[{"role":"user","content":"Hej JACO"}]}'

Integration alternative: import the exported modules from the package rather than starting a second HTTP server. JACO-CODE remains owner of process supervision, approval policy, memory, tasks and tools.

### Configuration

- JACO_MODEL_BASE_URL: upstream URL including /v1 (default: http://127.0.0.1:11435/v1).
- JACO_MODEL_ID: exact upstream model identifier; required when more than one is available.
- JACO_MODEL_API_KEY: optional token used for upstream requests only, never written in logs.
- JACO_MODEL_TIMEOUT_MS: HTTP timeout, default 300000 (Bonsai has very slow prompt processing on CPU).
- JACO_MODEL_RUNTIME_PORT: localhost HTTP port, default 3131.
- JACO_MODEL_THINKING: auto (default), off, on or provider. Auto sends `enable_thinking=false` for Bonsai only; other models are unchanged. Per-request `chat_template_kwargs.enable_thinking` always takes precedence.

On the Pixel 10 Pro, Bonsai-27B-Q1_0 has been observed at ~0.35 prompt tokens/s and ~0.37 generated tokens/s in CPU mode. `enable_thinking=false` allows short visible responses but **does not accelerate prompt processing**. CLI `chat` limits generation to 32 tokens by default to prevent runaway CPU tests. These observations are device-specific, not general benchmark claims.

### Commands

- ./start.sh serve: run HTTP adapter in foreground (default).
- ./start.sh status: fetch live runtime health over localhost.
- ./start.sh doctor: diagnose configured upstream without starting HTTP adapter.
- ./start.sh capabilities: inspect compile/integration truthfulness.
- ./start.sh chat "question": ask real configured model; fails if none is available.
- npm test: Node.js HTTP/adapter contract tests (mock provider is ONLY a test fixture).

### ACFQ research and benchmarks

The original C/Python kernel research is preserved, separate from runtime operations:

    pkg install clang python python-numpy
    ./build.sh
    python3 acfq_rig.py --synthetic
    python3 acfq_rig.py --data acts

- acfq_kernel.c: Q4/Q8 activation-sparse GEMV; NEON/fp16 paths.
- test_acfq.c: C numerical self-tests.
- acfq_rig.py: numerical accuracy, timing and approximate byte access.
- capture_activations.py: PyTorch/Transformers hooks for actual model activations, run on a capable computer.

Important provenance: files named X_*.npy and W_*.npy under acts/ may be generated artificially. Their existence does NOT prove real GGUF/LLM activations. Reading raw GGUF bytes as int8 is NOT a GGUF tensor decoder. Benchmark provenance remains unverified unless independently established. MB/tok is an estimated logical weight-read count, not measured DRAM traffic. Battery current APIs may return unreliable or negative net mJ/tok; do not use them as validated energy savings.

ACFQ activation requires proper GGUF tensor conversion, recorded REAL activations, model-level perplexity, end-to-end tokens/s against original llama.cpp/GGML, ARM verification, safe API validation and rollback. The experimental kernel is never silently enabled.

## Files

- runtime/provider.mjs: reusable real-model provider adapter and status.
- runtime/server.mjs: loopback HTTP adapter and capability declaration.
- runtime/cli.mjs: small operator interface without daemon supervision.
- runtime/runtime.test.mjs: HTTP contract tests.
- package.json: scripts and export contract.
- acfq_kernel.c, acfq_rig.py, capture_activations.py: kernel experiment.
- .github/workflows/test.yml: Node + C/NEON test workflow.

This repo is a model runtime COMPONENT, not the full JACO Cognitive Runtime.

## C++ Native Benchmark

Build with `sh bench/build.sh`, then run `./bench/native_probe --port 11435 --reps 2 --predict 4` while the **existing** llama.cpp model server runs. This measures actual llama.cpp `/completion` token throughput; it does **not** accelerate GGML or ACFQ. See [benchmark instructions](bench/README.md).

## Verified Gemma baseline and performance comparison

On 2026-10-18, Gemma 4 E2B through LiteRT-LM (loopback port 9379) and
JACO Model Runtime (loopback port 3131) returned a real answer to
`2+2`: `2+2 är 4.`. This is the stable functional reference.
**This does not mean inference is accelerated.** ACFQ has not been connected
to LiteRT/Gemma kernels and must remain off for production inference.

With both existing services running, compare direct and adapter calls:

```sh
cd ~/jaco-qwant-llm
node bench/gemma-compare.mjs --repetitions 3
```

The script alternates the order of identical sequential HTTP requests to
avoid competing model processes; it reports wall-clock timing, usage counts,
and adapter overhead. It does not alter JACO services or model files.
Check output length, prompt and completion token counts before inferring
throughput differences. Establish a repeatable baseline before modifying any
quantized tensor kernels. Real ACFQ integration requires a compatible tensor
representation, numerical-equivalence tests, full-model quality validation,
controlled end-to-end throughput comparison, and rollback on regressions.

## LiteRT engine compatibility check (not an engine monkeypatch)

The optional `runtime/engine_patch.py` is deliberately a **read-only probe**.
Run `python3 runtime/engine_patch.py` to inspect whether the local Python
LiteRT module and `libacfq.so` are present/loadable. Neither is proof of
kernel execution in Gemma. This module does not change `litert_lm.Engine`,
intercept `create_conversation()`, mutate installed packages, or alter the
running LiteRT server. A Python FFI call before inference cannot redirect
compiled LiteRT operator execution into ACFQ. Enabling a real replacement
requires identifying a supported custom operator/delegate interface,
matching tensor types and layouts, and passing numerical and end-to-end tests.

See [LiteRT source architecture review](research/README.md) for evidence-first native integration analysis.

## Hugging Face models and free public demo

JACO can use **Hugging Face Inference Providers** via the SAME model runtime.
The local Gemma 4 E2B installation remains the default; HF mode is opt-in.

```sh
# Create a fine-grained HF token with "Make calls to Inference Providers".
# Keep the token private. Never commit it to git or paste it into a public demo.
export HF_TOKEN="your-private-token"
# Clear local settings from the previous Gemma/LiteRT session:
unset JACO_MODEL_BASE_URL JACO_MODEL_ID JACO_MODEL_API_KEY
export JACO_MODEL_PROVIDER=huggingface
export JACO_MODEL_THINKING=provider
./start.sh models qwen
# Choose an exact ID from the returned inference model list.
export JACO_MODEL_ID="your-org/your-model"
./start.sh doctor
./start.sh chat "Svara kort: hej"
```

- `JACO_MODEL_PROVIDER=huggingface` selects only `https://router.huggingface.co/v1`.
- Only this explicit mode reads `HF_TOKEN`; custom and local providers never inherit it.
- `JACO_MODEL_ID` must be available through HF Inference Providers, not merely uploaded to Hub.
- `JACO_MODEL_API_KEY` may also be used, but no token is stored in JACO's source.
- HF inference can have usage charges and is subject to provider availability.
- Return to Android Gemma with `unset JACO_MODEL_PROVIDER JACO_MODEL_API_KEY; export JACO_MODEL_BASE_URL=http://127.0.0.1:9379/v1 JACO_MODEL_ID=gemma4-e2b`.

A **free static Hugging Face Space** demonstration lives in
[`demo/huggingface-space`](demo/huggingface-space/README.md).
It searches public Hub model cards in real time, never stores credentials, and
links to HF's hosted Playground for live inference. A public JACO-branded chat
with secret-funded inference requires a trusted rate-limited backend; the static
demo intentionally does not claim to run a model or expose a shared token.

### Publish the public Space

The `demo/huggingface-space` folder is deployable as a **static** Space.
The optional manual workflow `.github/workflows/huggingface-space-publish.yml`
publishes it without granting runtime or frontend access to your token.

1. Create a Hugging Face fine-grained **write token** with permission to create
   and/or upload to the target Space. It is separate from your inference token.
2. Put it in GitHub repository **Actions secret** `HF_SPACE_WRITE_TOKEN`.
3. Run the workflow **Publish JACO Hugging Face Space** with your desired
   `space_id` (e.g. `owner/jaco-model-explorer`). It creates a public Static
   Space if absent and uploads just `README.md` and `index.html`.
4. Visit the resulting Space. No paid compute, inference model or agent service
   is automatically deployed.

Alternative: create a public Static Space in the Hugging Face web UI and
upload those two files manually. The connected HF chat integration currently
has no write scope, so it cannot publish directly from this conversation.

## Evidence-backed model research and router selection

- `./start.sh recommend coding` — curated recommendations; no token or network
  needed. Other goals: `agents`, `local`, `multimodal`.
- `./start.sh rank agents` — in opt-in Hugging Face mode, inspect the actual
  router `/v1/models` provider metadata; keep only models with a **live**
  provider that explicitly advertises tool calling for agent jobs.
- Only a completed inference request with tests can prove generation quality.
  Ranking does not activate, download, pay for or run a model.
- Source: `demo/huggingface-space/catalog.json`, one canonical catalog
  read by both the CLI and the static public demo.
- Methodology and caveats: `research/HF_MODEL_REVIEW_2026-10-20.md`.

Be careful: `rank` uses your Hugging Face token but requests metadata only.
The public Space never receives credentials.
