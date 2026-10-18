// JACO Model Runtime: upstream adapter. Never fabricates model output.
export class RuntimeError extends Error {
  constructor(message, status = 502) {
    super(message);
    this.status = status;
  }
}
export function configFromEnv(env = process.env) {
  const raw = env.JACO_MODEL_BASE_URL || "http://127.0.0.1:11435/v1";
  let url;
  try { url = new URL(raw); } catch { throw new RuntimeError("Invalid JACO_MODEL_BASE_URL", 400); }
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash)
    throw new RuntimeError("Model URL must be HTTP(S) without credentials or query", 400);
  const timeoutMs = Number(env.JACO_MODEL_TIMEOUT_MS || 120000);
  const port = Number(env.JACO_MODEL_RUNTIME_PORT || 3131);
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 600000)
    throw new RuntimeError("Invalid JACO_MODEL_TIMEOUT_MS", 400);
  if (!Number.isInteger(port) || port < 1 || port > 65535)
    throw new RuntimeError("Invalid JACO_MODEL_RUNTIME_PORT", 400);
  return { baseUrl: url.toString().replace(/\/+$/, ""), model: env.JACO_MODEL_ID?.trim() || null,
    apiKey: env.JACO_MODEL_API_KEY || "", timeoutMs, port };
}
async function request(config, path, opts = {}, fetchImpl = fetch) {
  const timeout = AbortSignal.timeout(config.timeoutMs);
  const signal = opts.signal ? AbortSignal.any([opts.signal, timeout]) : timeout;
  try {
    return await fetchImpl(config.baseUrl + "/" + path, { ...opts, signal,
      headers: { ...(opts.body ? { "content-type": "application/json" } : {}),
        ...(config.apiKey ? { authorization: "Bearer " + config.apiKey } : {}) } });
  } catch (error) {
    throw new RuntimeError("Model provider unavailable or timeout: " + error.message, 502);
  }
}
export async function discoverModels(config, fetchImpl = fetch) {
  const res = await request(config, "models", {}, fetchImpl);
  if (!res.ok) throw new RuntimeError("Model discovery returned HTTP " + res.status, 502);
  let data;
  try { data = await res.json(); } catch { throw new RuntimeError("Model discovery returned invalid JSON", 502); }
  if (!Array.isArray(data?.data)) throw new RuntimeError("Model discovery returned no list", 502);
  return data.data.filter(x => typeof x?.id === "string" && x.id.length).map(x => x.id);
}
export function selectModel(config, models, explicit = null) {
  const model = explicit || config.model;
  if (model) {
    if (typeof model !== "string") throw new RuntimeError("Invalid model ID", 400);
    if (!models.includes(model)) throw new RuntimeError("Selected model absent from provider", 503);
    return model;
  }
  if (models.length === 1) return models[0];
  throw new RuntimeError(models.length ? "Set JACO_MODEL_ID: multiple models available" : "No models available", 503);
}
export async function inspectProvider(config, fetchImpl = fetch) {
  try {
    const availableModels = await discoverModels(config, fetchImpl);
    const model = selectModel(config, availableModels);
    return { state: "READY", reachable: true, model, availableModels,
      verification: "model-discovery-only" };
  } catch (error) {
    return { state: "DEGRADED", reachable: false, model: config.model,
      availableModels: [], reason: error.message, verification: "model-discovery-only" };
  }
}
export async function chatCompletion(config, payload, fetchImpl = fetch, signal) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload) ||
      !Array.isArray(payload.messages) || payload.messages.length === 0)
    throw new RuntimeError("Nonempty messages array required", 400);
  for (const msg of payload.messages) {
    if (!msg || !["system", "developer", "user", "assistant", "tool"].includes(msg.role) ||
        !(typeof msg.content === "string" || Array.isArray(msg.content) || msg.content === null))
      throw new RuntimeError("Invalid chat message", 400);
  }
  if (payload.model != null && (typeof payload.model !== "string" || !payload.model))
    throw new RuntimeError("Invalid model ID", 400);
  const model = payload.model || config.model || selectModel(config, await discoverModels(config, fetchImpl));
  return request(config, "chat/completions", { method: "POST",
    body: JSON.stringify({ ...payload, model }), signal }, fetchImpl);
}
