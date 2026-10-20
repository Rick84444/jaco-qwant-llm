import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const space = new URL("../demo/huggingface-space/index.html", import.meta.url);
const doc = new URL("../demo/huggingface-space/README.md", import.meta.url);

test("QWANT gets a prominent, accurately labeled product preview", () => {
  const html = readFileSync(space, "utf8");
  assert.match(html, /JACO QWANT LLM/);
  assert.match(html, /ACFQ.*ännu inte aktiverad/);
  assert.match(html, /inte nya publicerade modellvikter/);
  assert.match(html, /gemma4-e2b/);
  assert.match(html, /github\.com\/Rick84444\/jaco-qwant-llm/);
  assert.match(html, /api\.github\.com\/repos\/Rick84444\/jaco-qwant-llm\/actions\/runs/);
  // Shell examples may document variable names; never embed an actual credential.
  assert.doesNotMatch(html, /\b(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,})\b/);
  assert.doesNotMatch(html, /<script[^>]*>[\s\S]*?Bearer\s+hf_/i);
});
test("Hugging Face static README distinguishes runtime from trained weights", () => {
  const info=readFileSync(doc,"utf8");
  assert.match(info,/sdk: static/);
  assert.match(info,/no published/);
  assert.match(info,/NOT end-to-end Gemma inference speedups/);
  assert.match(info,/what visitors CANNOT/i);
});
