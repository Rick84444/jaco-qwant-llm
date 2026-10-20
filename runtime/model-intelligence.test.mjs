import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { curatedCatalog, recommendModels, rankLiveCandidates } from "./model-intelligence.mjs";
import { configFromEnv, discoverModelDetails } from "./provider.mjs";

test("model review includes supported primary cards and auditable sources", () => {
  const catalog=curatedCatalog();
  assert.equal(catalog.asOf,"2026-10-20");
  assert.ok(catalog.models.length >= 6);
  assert.equal(new Set(catalog.models.map(m=>m.id)).size,catalog.models.length);
  for (const m of catalog.models) {
    assert.match(m.source,/^https:\/\/huggingface\.co\//);
    assert.equal(m.source,"https://huggingface.co/"+m.id);
    for (const goal of m.focus) assert.ok(catalog.goals.includes(goal));
    for (const n of Object.values(m.fit)) assert.ok(Number.isInteger(n) && n >= 1 && n <= 5);
  }
  const local=recommendModels("local");
  assert.equal(local[0].id,"google/gemma-4-E2B");
  assert.equal(local[0].providerVerified,false);
  assert.equal(local[0].evaluatedInJaco,false);
  assert.throws(()=>recommendModels("unknown"),/Goal must be/);
});
test("provider discovery preserves router metadata without claiming a generation",async()=>{
  const cfg=configFromEnv({JACO_MODEL_PROVIDER:"huggingface",HF_TOKEN:"placeholder"});
  const fake=async(url)=> {
    assert.ok(url.endsWith("/models"));
    return new Response(JSON.stringify({data:[{id:"Qwen/Qwen3-Coder-Next",
      providers:[{provider:"a",status:"live",supports_tools:true,pricing:{input:1.2,output:2.4}}]}]}),{status:200});
  };
  const models=await discoverModelDetails(cfg,fake);
  assert.equal(models[0].providers[0].supports_tools,true);
  assert.equal(rankLiveCandidates(models,"agents")[0].providerVerified,true);
});
test("agent recommendations reject absent, down and unknown tool support",()=>{
 const catalog=[
  {id:"Qwen/Qwen3-Coder-Next",providers:[
   {provider:"down",status:"error",supports_tools:true},
   {provider:"unknown",status:"live"},
   {provider:"no-tools",status:"live",supports_tools:false}]},
  {id:"moonshotai/Kimi-K3",providers:[{provider:"good",status:"live",supports_tools:true,pricing:{input:0.8,output:1.4}}]}
 ];
 const ranked=rankLiveCandidates(catalog,"agents");
 assert.deepEqual(ranked.map(m=>m.id),["moonshotai/Kimi-K3"]);
 assert.equal(ranked[0].activeProviders[0].inputUSDPerMillion,0.8);
 assert.equal(ranked[0].evaluatedInJaco,false);
 assert.deepEqual(rankLiveCandidates([], "coding"),[]);
});
test("public demo has no embedded token or fake generation, and model check is opt-in",()=>{
 const html=readFileSync(new URL("../demo/huggingface-space/index.html",import.meta.url),"utf8");
 assert.match(html,/\.\/catalog\.json/);
 assert.match(html,/Kontrollera provider/);
 assert.match(html,/router\.huggingface\.co\/v1\/models/);
 // Documentation of environment variable names is safe; literal secrets are not.
 assert.doesNotMatch(html,/\b(?:hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,})\b/);
 assert.doesNotMatch(html,/<script[^>]*>[\s\S]*?Bearer\s+hf_/i);
});
