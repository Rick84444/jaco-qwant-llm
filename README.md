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

The default runtime listens ONLY on localhost port 3131. Ctrl+C stops it. No automatic installs and no unprotected port 8000 server.

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
- JACO_MODEL_TIMEOUT_MS: HTTP timeout, default 120000.
- JACO_MODEL_RUNTIME_PORT: localhost HTTP port, default 3131.

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
