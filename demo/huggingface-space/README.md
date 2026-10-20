---
title: JACO Aura and QWANT Preview
emoji: 🧠
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: JACO QWANT runtime and Aura model explorer
---

# JACO Model Explorer

A static, public, no-token demo showing the live public Hugging Face Hub model
catalog. It links to model cards and to Hugging Face Inference Playground.

**Important:** a model card is not a running inference provider. This public
Space has no private inference credential and does not execute JACO, Gemma or
ACFQ. It is the first safe public demo component. A true JACO chat needs
a separately deployed, authenticated backend with rate limiting and billing
controls; do not expose a shared HF token to a browser.

## Publish

Create a public **Static** Space in your Hugging Face account. Upload both
`README.md` and `index.html` to the root of that Space. It needs no
secrets, servers, API keys or paid compute. You can also run `index.html`
locally in a browser.

The model list is fetched from `https://huggingface.co/api/models`.
No fake replies or hard-coded inference success statuses are displayed.

## Model Intelligence review (2026-10-20)

The `catalog.json` file supplies a transparent shortlist for coding, agent
workflows, local inference and multimodality. Ratings are **editorial task fit
only** and are not benchmark scores. Clicking "Kontrollera provider" makes a
public read-only request to the Hugging Face router's model-metadata endpoint;
the browser may not be permitted to read it (CORS). Unknown/unavailable
metadata never becomes a claim that a model was called or is free to use.

The demo also searches the full public Hub. Models on the Hub are not
necessarily served by an inference provider. Hosted inference may be billed.

## JACO QWANT LLM – official technical showcase

This public Space presents **JACO QWANT** prominently as the standalone
model-runtime adapter and experimental ACFQ native INT4 research kernel.

What visitors can really test:
- Clone the public QWANT repo on a local device with Node.js and a pre-existing
  OpenAI-compatible model server; run `doctor` and a real `chat` request.
- Read reproducible CI, kernel benchmarks and decode graph evidence.
- Access the JACO IDE product and inspect model cards in this Space.

What visitors CANNOT do in this free static Space:
- Execute QWANT Gemma locally on Hugging Face's static hosting.
- Run JACO's private autonomous IDE/runtime from this public front end.
- Benchmark ACFQ acceleration on a complete language model.
- Download an original trained "QWANT language model": there are no published
  independent QWANT model weights in the repository.

ACFQ experiments use synthetic activation/weight matrices. Their speedups
against a scalar reference are NOT end-to-end Gemma inference speedups.
The public Space should upgrade the QWANT label from "development" to
"integrated model runtime" only when isolated Android execution, numerical
parity and measured tokens/s have passing reproducible evidence.

Privacy: the Space is read-only, has no shared tokens and does not install or
run code on its visitors' devices.
