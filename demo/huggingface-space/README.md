---
title: JACO Model Explorer
emoji: 🧠
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Explore open models for JACO without sharing access tokens
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
