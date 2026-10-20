import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { configFromEnv, discoverModels, selectModel, chatCompletion } from "./provider.mjs";

test("Hugging Face is opt-in and does not inherit HF_TOKEN locally", () => {
  const cfg = configFromEnv({ HF_TOKEN: "private" });
  assert.equal(cfg.provider, "custom");
  assert.equal(cfg.apiKey, "");
  assert.equal(cfg.baseUrl, "http://127.0.0.1:11435/v1");
});
test("Hugging Face mode uses only the canonical HTTPS router", () => {
  const cfg = configFromEnv({ JACO_MODEL_PROVIDER: "huggingface", HF_TOKEN: "private" });
  assert.equal(cfg.baseUrl, "https://router.huggingface.co/v1");
  assert.equal(cfg.apiKey, "private");
  for (const bad of [
    "https://other.example/v1", "http://router.huggingface.co/v1",
    "https://router.huggingface.co/v2", "https://router.huggingface.co/v1/extra"
  ]) assert.throws(() => configFromEnv({
    JACO_MODEL_PROVIDER: "huggingface", HF_TOKEN: "private", JACO_MODEL_BASE_URL: bad
  }), /only permits/);
  assert.throws(() => configFromEnv({ JACO_MODEL_PROVIDER: "huggingface" }), /HF_TOKEN/);
  assert.throws(() => configFromEnv({ JACO_MODEL_PROVIDER: "unknown" }), /JACO_MODEL_PROVIDER/);
});
test("HF model discovery uses bearer auth and respects listed model IDs", async () => {
  const cfg = configFromEnv({ JACO_MODEL_PROVIDER: "huggingface", HF_TOKEN: "private",
    JACO_MODEL_ID: "Qwen/example:cheapest" });
  const seen = [];
  const fakeFetch = async (url, init) => {
    seen.push({ url, init });
    if (url.endsWith("/models"))
      return new Response(JSON.stringify({ data: [{ id: "Qwen/example" }] }), { status: 200 });
    return new Response(JSON.stringify({ choices: [{ message: { content: "OK" } }] }), { status: 200 });
  };
  const available = await discoverModels(cfg, fakeFetch);
  assert.deepEqual(available, ["Qwen/example"]);
  assert.equal(selectModel(cfg, available), "Qwen/example:cheapest");
  assert.throws(() => selectModel(cfg, available, "unknown/model"), /absent/);
  const reply = await chatCompletion(cfg, { messages: [{ role: "user", content: "Hello" }] }, fakeFetch);
  assert.equal((await reply.json()).choices[0].message.content, "OK");
  assert.equal(JSON.parse(seen[1].init.body).model, "Qwen/example:cheapest");
  for (const req of seen) {
    assert.equal(req.init.headers.authorization, "Bearer private");
    assert.ok(req.url.startsWith("https://router.huggingface.co/v1/"));
  }
});
test("static demo is explicitly a catalog, not a fake inference server", () => {
  const html = readFileSync(new URL("../demo/huggingface-space/index.html", import.meta.url), "utf8");
  assert.match(html, /huggingface\.co\/api\/models/);
  assert.match(html, /huggingface\.co\/playground/);
  // Shell examples may document variable names; never embed an actual credential.
  assert.doesNotMatch(html, /\b(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,})\b/);
  assert.doesNotMatch(html, /<script[^>]*>[\s\S]*?Bearer\s+hf_/i);
});
