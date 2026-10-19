import test from "node:test";
import assert from "node:assert/strict";
import {parse,run,summarize,measure} from "./gemma-compare.mjs";
test("bounded comparison arguments",()=>{assert.equal(parse([]).model,"gemma4-e2b");assert.throws(()=>parse(["--repetitions","0"]),/Invalid/);});
test("same inference request on both paths with alternating order",async()=>{
 const calls=[];
 const mock=async(url,opts)=>{calls.push([url,JSON.parse(opts.body)]);return {ok:true,json:async()=>({choices:[{message:{content:"Vatten."}}],usage:{prompt_tokens:12,completion_tokens:3}})};};
 const value=await run({...parse(["--repetitions","2"])},mock);
 assert.equal(value.records.length,4);assert.deepEqual(calls.map(x=>x[0]),[parse([]).direct,parse([]).adapter,parse([]).adapter,parse([]).direct]);
 for(const call of calls)assert.deepEqual(call[1],calls[0][1]);assert.equal(value.summary.adapterMinusDirectMs!==undefined,true);
});
test("no false generation success",async()=>{
 await assert.rejects(measure("http://127.0.0.1",{},3000,async()=>({ok:true,json:async()=>({choices:[]})})),/No generated/);
 assert.equal(summarize([{route:"direct",elapsedMs:10},{route:"adapter",elapsedMs:15}]).adapterMinusDirectMs,5);
});

test("comparison rejects missing or malformed timing data",()=>{
 assert.throws(()=>summarize([]),/Both routes/);
 assert.throws(()=>summarize([{route:"direct",elapsedMs:11}]),/Both routes/);
 assert.throws(()=>summarize([{route:"direct",elapsedMs:NaN},
   {route:"adapter",elapsedMs:2}]),/Both routes/);
});
test("provider HTTP failures are reported before JSON decoding",async()=>{
 await assert.rejects(measure("http://127.0.0.1",{},3000,
   async()=>({ok:false,status:503,json:async()=>{throw Error("not JSON");}})),
   /Upstream HTTP 503/);
});
