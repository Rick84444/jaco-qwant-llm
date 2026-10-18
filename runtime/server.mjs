// Embedded loopback adapter usable directly or as a standalone component.
import http from "node:http";
import { Readable } from "node:stream";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { configFromEnv, inspectProvider, discoverModels, chatCompletion, RuntimeError } from "./provider.mjs";
const library = resolve(dirname(fileURLToPath(import.meta.url)), "..", "libacfq.so");
export function capabilities() {
  return { name: "jaco-model-runtime", contractVersion: 1,
    inference: { provider: "openai-compatible", generationTested: false },
    acfq: { libraryBuilt: existsSync(library), inferenceIntegrated: false, enabled: false,
      mode: "experimental-benchmark-only" } };
}
function json(res, status, object) {
  const value = Buffer.from(JSON.stringify(object));
  res.writeHead(status, { "content-type": "application/json; charset=utf-8",
    "content-length": value.length, "cache-control": "no-store", "x-content-type-options": "nosniff" });
  res.end(value);
}
async function parseBody(req) {
  const parts = [];
  let size = 0;
  for await (const part of req) {
    size += part.length;
    if (size > 1024 * 1024) throw new RuntimeError("Body exceeds 1 MiB", 413);
    parts.push(part);
  }
  try { return JSON.parse(Buffer.concat(parts).toString("utf8")); }
  catch { throw new RuntimeError("Invalid JSON", 400); }
}
export function createRuntimeServer(config = configFromEnv(), fetchImpl = fetch) {
  return http.createServer(async (req, res) => {
    const path = (req.url || "").split("?")[0];
    if (req.method === "GET" && path === "/capabilities") return json(res, 200, capabilities());
    if (req.method === "GET" && path === "/health") {
      const state = await inspectProvider(config, fetchImpl);
      return json(res, state.state === "READY" ? 200 : 503,
        { system: "JACO", component: "model-runtime", ...state, acfqInferenceEnabled: false });
    }
    if (req.method === "GET" && path === "/v1/models") {
      try {
        const models = await discoverModels(config, fetchImpl);
        return json(res, 200, { object: "list", data: models.map(id => ({ id, object: "model" })) });
      } catch (error) { return json(res, error.status || 502, { error: { message: error.message } }); }
    }
    if (req.method === "POST" && path === "/v1/chat/completions") {
      const controller = new AbortController();
      res.on("close", () => { if (!res.writableFinished) controller.abort(); });
      try {
        const payload = await parseBody(req);
        const upstream = await chatCompletion(config, payload, fetchImpl, controller.signal);
        res.writeHead(upstream.status, { "content-type": upstream.headers.get("content-type") || "application/json",
          "cache-control": "no-store", "x-content-type-options": "nosniff" });
        if (!upstream.body) return res.end();
        Readable.fromWeb(upstream.body).on("error", () => res.destroy()).pipe(res);
      } catch (error) {
        if (!res.headersSent) return json(res, error instanceof RuntimeError ? error.status : 502,
          { error: { message: error.message || "Provider error" } });
        res.destroy();
      }
      return;
    }
    return json(res, 404, { error: { message: "Unknown endpoint" } });
  });
}
export async function serve(config = configFromEnv()) {
  const server = createRuntimeServer(config);
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(config.port, "127.0.0.1", resolve);
  });
  console.log("JACO Model Runtime: http://127.0.0.1:" + config.port);
  console.log("ACFQ is benchmark-only, NOT integrated into model inference.");
  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.once(signal, () => server.close(() => process.exit(0)));
  }
  return server;
}
