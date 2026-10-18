import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { once } from "node:events";
import { configFromEnv, discoverModels, inspectProvider, chatCompletion, selectModel } from "./provider.mjs";
import { createRuntimeServer, capabilities } from "./server.mjs";

async function mock(handler) {
  const server = http.createServer(handler);
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return { base: "http://127.0.0.1:" + server.address().port + "/v1",
    close: () => new Promise(done => server.close(done)) };
}
function send(res, status, data) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(data));
}
test("no falsely READY status if upstream unavailable", async () => {
  const upstream = await mock((req,res) => send(res, 503, { error: "down" }));
  try {
    const state = await inspectProvider(configFromEnv({ JACO_MODEL_BASE_URL: upstream.base }));
    assert.equal(state.state, "DEGRADED");
  } finally { await upstream.close(); }
});
test("real model discovery and ambiguous selection", async () => {
  const upstream = await mock((req,res) => send(res, 200, { data: [{ id: "local-model" }] }));
  try {
    const cfg = configFromEnv({ JACO_MODEL_BASE_URL: upstream.base });
    assert.deepEqual(await discoverModels(cfg), ["local-model"]);
    assert.equal((await inspectProvider(cfg)).model, "local-model");
    assert.throws(() => selectModel(cfg, ["a", "b"]), /multiple models/);
  } finally { await upstream.close(); }
});
test("HTTP compatibility and SSE forward real upstream payload", async () => {
  let handled = 0;
  const upstream = await mock(async (req,res) => {
    if (req.url === "/v1/models") return send(res, 200, { data: [{ id: "local-model" }] });
    if (req.url === "/v1/chat/completions") {
      let body = "";
      for await (const p of req) body += p;
      const payload = JSON.parse(body);
      assert.equal(payload.model, "local-model");
      assert.equal(payload.messages[0].content, "Hej");
      handled++;
      if (payload.stream) {
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.end('data: {"choices":[{"delta":{"content":"hej"}}]}\n\ndata: [DONE]\n\n');
      } else send(res, 200, { choices: [{ message: { content: "Model answer" } }] });
      return;
    }
    send(res, 404, {});
  });
  const app = createRuntimeServer(configFromEnv({ JACO_MODEL_BASE_URL: upstream.base }));
  app.listen(0, "127.0.0.1");
  await once(app, "listening");
  const base = "http://127.0.0.1:" + app.address().port;
  try {
    const health = await fetch(base + "/health");
    assert.equal(health.status, 200);
    assert.equal((await health.json()).verification, "model-discovery-only");
    const bad = await fetch(base + "/v1/chat/completions", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: [] })
    });
    assert.equal(bad.status, 400);
    const body = { messages: [{ role: "user", content: "Hej" }] };
    const response = await fetch(base + "/v1/chat/completions", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body)
    });
    assert.equal((await response.json()).choices[0].message.content, "Model answer");
    const streaming = await fetch(base + "/v1/chat/completions", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...body, stream: true })
    });
    assert.match(await streaming.text(), /data: \[DONE\]/);
    assert.equal(handled, 2);
    assert.equal(capabilities().acfq.inferenceIntegrated, false);
  } finally {
    await new Promise(done => app.close(done));
    await upstream.close();
  }
});
test("chat input checked before any request", async () => {
  const cfg = configFromEnv({ JACO_MODEL_ID: "m" });
  await assert.rejects(chatCompletion(cfg, { messages: [] }), /Nonempty/);
  await assert.rejects(chatCompletion(cfg, { messages: [{ role: "invalid", content: "x" }] }), /Invalid chat/);
});

test("Bonsai defaults to direct answers while preserving explicit thinking requests", async () => {
  const requests = [];
  const upstream = await mock(async (req,res) => {
    if (req.url === "/v1/models") return send(res, 200, { data: [{ id: "Bonsai-27B-Q1_0.gguf" }] });
    let body = "";
    for await (const p of req) body += p;
    requests.push(JSON.parse(body));
    send(res, 200, { choices: [{ message: { content: "4" } }] });
  });
  try {
    const cfg = configFromEnv({ JACO_MODEL_BASE_URL: upstream.base });
    const base = { messages: [{ role: "user", content: "2+2=" }] };
    await chatCompletion(cfg, base);
    await chatCompletion(cfg, { ...base, chat_template_kwargs: { enable_thinking: true } });
    await chatCompletion(configFromEnv({
      JACO_MODEL_BASE_URL: upstream.base, JACO_MODEL_THINKING: "on"
    }), base);
    await chatCompletion(configFromEnv({
      JACO_MODEL_BASE_URL: upstream.base, JACO_MODEL_THINKING: "provider"
    }), base);
    assert.equal(requests.length, 4);
    assert.equal(requests[0].chat_template_kwargs.enable_thinking, false);
    assert.equal(requests[1].chat_template_kwargs.enable_thinking, true);
    assert.equal(requests[2].chat_template_kwargs.enable_thinking, true);
    assert.equal(requests[3].chat_template_kwargs, undefined);
  } finally { await upstream.close(); }
});

test("thinking modes do not modify other provider requests unless requested", async () => {
  let upstreamBody;
  const upstream = await mock(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    upstreamBody = JSON.parse(body);
    send(res, 200, { choices: [{ message: { content: "ok" } }] });
  });
  try {
    await chatCompletion(configFromEnv({
      JACO_MODEL_BASE_URL: upstream.base, JACO_MODEL_ID: "another-model"
    }), { messages: [{ role: "user", content: "hej" }] });
    assert.equal(upstreamBody.chat_template_kwargs, undefined);
    await chatCompletion(configFromEnv({
      JACO_MODEL_BASE_URL: upstream.base, JACO_MODEL_ID: "another-model",
      JACO_MODEL_THINKING: "off"
    }), { messages: [{ role: "user", content: "hej" }] });
    assert.equal(upstreamBody.chat_template_kwargs.enable_thinking, false);
    assert.throws(() => configFromEnv({ JACO_MODEL_THINKING: "sometimes" }), /must be/);
  } finally { await upstream.close(); }
});
