#!/usr/bin/env python3
"""Evidence-driven LiteRT native path investigation. Read-only."""
import argparse
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

BASE="https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM"
TARGETS={
 "executor_settings":"runtime/executor/llm_executor_settings.h",
 "executor_settings_impl":"runtime/executor/llm_executor_settings.cc",
 "executor":"runtime/executor/litert_compiled_model_executor.cc",
 "engine_c_api":"c/litert_lm_c_api.cc",
 "cpu_executor":"runtime/executor/llm_executor.cc",
}
QUESTIONS={
 "backend_dispatch":{"needles":("enable_ynnpack","Backend::CPU","CpuConfig"),
   "task":"Trace CPU executor creation and delegation to XNNPACK/YNNPACK; locate callsites, not just setters."},
 "native_op_registration":{"needles":("delegate","custom_op","registration"),
   "task":"Find supported operator registration with ABI, lifecycle and fallback. Strings alone do NOT prove a hook."},
 "quantized_matmul":{"needles":("fully_connected","matmul","quantiz"),
   "task":"Identify Gemma E2B actual weight/activation types and tensor layout from the installed .litertlm graph."},
 "rollback":{"needles":("create","backend","status"),
   "task":"Require an independent opt-in native build and exact off-switch preserving Gemma on 9379."},
}
def report(sources,revision,installed_sha=None):
    evidence=[]
    for name,path in TARGETS.items():
        data=sources.get(name)
        if data is None:
            evidence.append({"source":path,"status":"not-retrieved","sha256":None,"matches":[]})
            continue
        lines=data.splitlines()
        found=[]
        for group,query in QUESTIONS.items():
            for word in query["needles"]:
                positions=[i for i,line in enumerate(lines,1) if re.search(re.escape(word),line,re.I)]
                if positions:
                    found.append({"area":group,"term":word,"line":positions[0],"excerpt":lines[positions[0]-1].strip()[:240]})
        evidence.append({"source":path,"status":"retrieved","sha256":hashlib.sha256(data.encode()).hexdigest(),"matches":found})
    actions=[]
    for group,query in QUESTIONS.items():
        refs=[{"source":r["source"],"line":m["line"],"term":m["term"]} for r in evidence for m in r["matches"] if m["area"]==group]
        actions.append({"area":group,"status":"candidate-locations-only" if refs else "unresolved",
            "candidate_locations":refs[:15],"required_action":query["task"],"proof_level":"lexical"})
    actions.append({"area":"version_parity","status":"unresolved",
        "required_action":"Compare installed LiteRT-LM wheel build metadata and native library version with pinned source; raw library SHA does not establish matching source revision.",
        "installed_library_sha256":installed_sha})
    actions.append({"area":"numeric_quality_performance","status":"unresolved",
        "required_action":"Compare exact and sparse kernel across real tensor samples, then PPL and end-to-end paired Gemma throughput against 6.03 s/60-token reference; reject regression or unsupported layout."})
    return {"schema":1,"upstream_revision":revision,"sources":evidence,
      "review":{"actor":"deterministic evidence auditor (not AI subagents)",
        "decision":"BLOCK_NATIVE_ACFQ","actions":actions},
      "inference_changed":False}
def fetch(revision):
    if not re.fullmatch("[a-f0-9]{40}",revision): raise ValueError("Pin full upstream SHA")
    src={}
    for name,path in TARGETS.items():
        try:
            request=urllib.request.Request(f"{BASE}/{revision}/{path}",headers={"User-Agent":"JACO-native-review"})
            with urllib.request.urlopen(request,timeout=25) as rsp: raw=rsp.read(1_500_001)
            if len(raw)>1_500_000:raise ValueError("source too large")
            src[name]=raw.decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code !=404:raise
    return src
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--revision",required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch",action="store_true")
    group.add_argument("--source-dir",type=Path)
    p.add_argument("--native-library",type=Path)
    p.add_argument("--output",type=Path,default=Path("bench/litert-native-review.json"))
    a=p.parse_args()
    if not re.fullmatch("[a-f0-9]{40}",a.revision):p.error("Must pin full upstream SHA")
    if a.fetch: sources=fetch(a.revision)
    else: sources={key:(a.source_dir/path).read_text() for key,path in TARGETS.items() if (a.source_dir/path).is_file()}
    libhash=None
    if a.native_library:
        h=hashlib.sha256()
        with a.native_library.open("rb") as f:
            for buf in iter(lambda:f.read(1024*1024),b""):h.update(buf)
        libhash=h.hexdigest()
    result=report(sources,a.revision,libhash)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(f"Wrote {a.output}: {sum(x['status']=='retrieved' for x in result['sources'])}/{len(TARGETS)} native files; ACFQ remains disabled.")
if __name__=="__main__":main()
