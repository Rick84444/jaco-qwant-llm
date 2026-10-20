#!/usr/bin/env node
import { configFromEnv, inspectProvider, chatCompletion, discoverModels } from "./provider.mjs";
import { capabilities, serve } from "./server.mjs";
const [command = "serve", ...args] = process.argv.slice(2);
async function main() {
  const cfg = configFromEnv();
  if (command === "serve") return serve(cfg);
  if (command === "capabilities") return console.log(JSON.stringify(capabilities(), null, 2));
  if (command === "doctor") {
    const result = await inspectProvider(cfg);
    console.log(JSON.stringify({ component: "jaco-model-runtime", ...result, capabilities: capabilities() }, null, 2));
    if (result.state !== "READY") process.exitCode = 2;
    return;
  }
  if (command === "status") {
    try {
      const r = await fetch("http://127.0.0.1:" + cfg.port + "/health", { signal: AbortSignal.timeout(4000) });
      console.log(await r.text());
      if (!r.ok) process.exitCode = 2;
    } catch {
      console.error("Model Runtime is not listening on localhost:" + cfg.port);
      process.exitCode = 2;
    }
    return;
  }
  if (command === "models") {
    const all = await discoverModels(cfg);
    const query = args.join(" ").toLowerCase().trim();
    const matches = query ? all.filter(id => id.toLowerCase().includes(query)) : all;
    console.log(JSON.stringify({ provider: cfg.provider, total: all.length,
      matching: matches.length, models: matches.slice(0, 40),
      truncated: matches.length > 40 }, null, 2));
    return;
  }
  if (command === "chat") {
    const prompt = args.join(" ").trim();
    if (!prompt) throw new Error('Usage: ./start.sh chat "Hej JACO"');
    const status = await inspectProvider(cfg);
    if (status.state !== "READY") throw new Error(status.reason);
    const result = await chatCompletion(cfg, {
      model: status.model, stream: false, max_tokens: 32, messages: [{ role: "user", content: prompt }]
    });
    if (!result.ok) throw new Error("Upstream model returned HTTP " + result.status);
    const value = await result.json();
    const answer = value?.choices?.[0]?.message?.content;
    if (typeof answer !== "string" || !answer.trim()) throw new Error("Upstream returned no usable generated text");
    console.log(answer);
    return;
  }
  console.error("Usage: ./start.sh [serve|status|doctor|capabilities|models [QUERY]|chat TEXT]");
  process.exitCode = 2;
}
main().catch(err => { console.error("JACO Model Runtime: " + err.message); process.exitCode = 1; });
