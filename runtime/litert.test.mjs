import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { once } from "node:events";
import { configFromEnv, inspectProvider, chatCompletion } from "./provider.mjs";
import { createRuntimeServer } from "./server.mjs";

test("LiteRT Gemma 4 E2B discovery and completion are passed through unchanged", async () => {
  const seen = [];
  const upstream = http.createServer(async (req, res) => {
    res.setHeader("content-type", "application/json");
    if (req.url === "/v1/models") return res.end(JSON.stringify({
      object: "list", data: [{ id: "gemma4-e2b", object: "model", owned_by: "litert-lm" }]
    }));
    if (req.url === "/v1/chat/completions") {
      let raw = "";
      for await (const chunk of req) raw += chunk;
      seen.push(JSON.parse(raw));
      return res.end(JSON.stringify({
        model: "gemma4-e2b", choices: [{ message: { role: "assistant", content: "GEMMA_CONNECTED" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 11, completion_tokens: 5 }
      }));
    }
    res.statusCode = 404; res.end("{}");
  });
  upstream.listen(0, "127.0.0.1");
  await once(upstream, "listening");
  const config = configFromEnv({
    JACO_MODEL_BASE_URL: `http://127.0.0.1:${upstream.address().port}/v1`,
    JACO_MODEL_ID: "gemma4-e2b",
    JACO_MODEL_THINKING: "provider"
  });
  const gateway = createRuntimeServer(config);
  gateway.listen(0, "127.0.0.1");
  await once(gateway, "listening");
  try {
    const status = await inspectProvider(config);
    assert.equal(status.state, "READY");
    assert.equal(status.model, "gemma4-e2b");
    assert.equal(status.verification, "model-discovery-only");
    const query = { messages: [{ role: "user", content: "Ping" }], max_tokens: 20, temperature: 0 };
    const direct = await chatCompletion(config, query);
    assert.equal((await direct.json()).choices[0].message.content, "GEMMA_CONNECTED");
    const res = await fetch(`http://127.0.0.1:${gateway.address().port}/v1/chat/completions`, {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(query)
    });
    assert.equal(res.status, 200);
    assert.equal((await res.json()).usage.completion_tokens, 5);
    assert.equal(seen.length, 2);
    for (const payload of seen) {
      assert.equal(payload.model, "gemma4-e2b");
      assert.equal(payload.chat_template_kwargs, undefined);
      assert.deepEqual(payload.messages, query.messages);
      assert.equal(payload.max_tokens, 20);
    }
  } finally {
    await new Promise(resolve => gateway.close(resolve));
    await new Promise(resolve => upstream.close(resolve));
  }
});
