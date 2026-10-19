#!/usr/bin/env node
// Sequential A/B measurement of real Gemma LiteRT inference, not ACFQ acceleration.
import { performance } from "node:perf_hooks";
const defaults = { direct:"http://127.0.0.1:9379/v1/chat/completions", adapter:"http://127.0.0.1:3131/v1/chat/completions", model:"gemma4-e2b", repetitions:3, timeoutMs:120000 };
export function parse(args) {
  const cfg={...defaults}; const names={"--repetitions":"repetitions","--timeout-ms":"timeoutMs"};
  for(let i=0;i<args.length;i+=2) {
    if(!names[args[i]]||!args[i+1]) throw Error("Usage: node bench/gemma-compare.mjs [--repetitions 3] [--timeout-ms 120000]");
    cfg[names[args[i]]]=Number(args[i+1]);
  }
  if(!Number.isInteger(cfg.repetitions)||cfg.repetitions<1||cfg.repetitions>10||!Number.isInteger(cfg.timeoutMs)||cfg.timeoutMs<1000||cfg.timeoutMs>600000) throw Error("Invalid parameters");
  return cfg;
}
export function summarize(rows) {
  if (!Array.isArray(rows) || !rows.length ||
      rows.some(r => !["direct", "adapter"].includes(r.route) ||
                 !Number.isFinite(r.elapsedMs) || r.elapsedMs < 0) ||
      !rows.some(r => r.route === "direct") ||
      !rows.some(r => r.route === "adapter"))
    throw Error("Both routes require finite, nonnegative timing observations");
  const median = route => { const xs=rows.filter(r=>r.route===route).map(r=>r.elapsedMs).sort((a,b)=>a-b);
    return (xs[(xs.length-1)>>1]+xs[xs.length>>1])/2; };
  const direct=median("direct"), adapter=median("adapter");
  return { directMedianMs:direct, adapterMedianMs:adapter, adapterMinusDirectMs:adapter-direct,
    note:"Wall-clock measurements only. No custom kernel acceleration. Compare token counts and output lengths." };
}
export async function measure(url,payload,timeoutMs,fetchImpl=fetch) {
  const start=performance.now();
  const res=await fetchImpl(url,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(payload),signal:AbortSignal.timeout(timeoutMs)});
  if(!res.ok) throw Error("Upstream HTTP "+res.status);
  const data=await res.json();
  if(typeof data?.choices?.[0]?.message?.content!=="string"||!data.choices[0].message.content.length) throw Error("No generated content");
  return {elapsedMs:Math.round(performance.now()-start),promptTokens:data.usage?.prompt_tokens??null,
    completionTokens:data.usage?.completion_tokens??null,outputLength:data.choices[0].message.content.length};
}
export async function run(cfg,fetchImpl=fetch) {
  const body={model:cfg.model,temperature:0,max_tokens:64,stream:false,
    messages:[{role:"user",content:"Skriv tre korta meningar om vatten på svenska."}]};
  const records=[];
  for(let round=1;round<=cfg.repetitions;round++) {
    for(const route of (round%2?["direct","adapter"]:["adapter","direct"])) {
      records.push({round,route,...await measure(cfg[route],body,cfg.timeoutMs,fetchImpl)});
    }
  }
  return {model:cfg.model,records,summary:summarize(records)};
}
if(process.argv[1]?.endsWith("/gemma-compare.mjs")) {
  run(parse(process.argv.slice(2))).then(x=>console.log(JSON.stringify(x,null,2))).catch(e=>{console.error("Comparison failed: "+e.message);process.exitCode=1;});
}
