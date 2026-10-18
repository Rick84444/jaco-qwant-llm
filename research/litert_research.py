#!/usr/bin/env python3
"""Offline, evidence-first LiteRT-LM source review; no runtime changes."""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT="https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM"
FILES={"engine":"python/litert_lm/engine.py","ffi":"python/litert_lm/_ffi.py","backend":"python/litert_lm/interfaces.py","android":"docs/getting-started/build-and-run.md"}
CHECKS={"engine":{"engine_creation":"litert_lm_engine_create","conversations":"create_conversation","ynnpack":"enable_ynnpack"},
        "ffi":{"loader":"liblitert-lm","binding":"ctypes"},
        "backend":{"backend_types":"class Backend"},
        "android":{"android_build":"android_arm64","bazel":"bazel"}}
GATES=("source_revision_matches_installed","native_operator_hook","tensor_layout_and_quantization",
       "numerical_equivalence","language_model_quality","real_inference_speedup","rollback")
def assess(sources,revision):
    evidence=[]
    for area,searches in CHECKS.items():
        lines=sources.get(area,"").splitlines()
        for label,needle in searches.items():
            matches=[(n,line.strip()) for n,line in enumerate(lines,1) if needle.lower() in line.lower()]
            evidence.append({"area":area,"check":label,"found":bool(matches),
                "line":matches[0][0] if matches else None,
                "excerpt":matches[0][1][:200] if matches else None})
    findings=[{"id":f"{e['area']}:{e['check']}","severity":"needs-evidence","action":"Inspect upstream source and version"}
              for e in evidence if not e["found"]]
    findings.extend({"id":gate,"severity":"blocker","action":"Require native source evidence and reproducible validation"}
                    for gate in GATES)
    return {"schema":1,"upstream_revision":revision,
        "files":{area:{"path":FILES[area],"sha256":hashlib.sha256(content.encode()).hexdigest()}
                 for area,content in sources.items()},
        "evidence":evidence,"senior_review":{"decision":"BLOCK_AFCQ_INFERENCE","findings":findings,
        "method":"deterministic verification gate; NOT autonomous senior-agent assessment"},
        "runtime_modified":False,"acfq_inference_enabled":False}
def retrieve(revision):
    if not re.fullmatch(r"[0-9a-f]{40}",revision):
        raise ValueError("A full pinned upstream commit SHA is required")
    result={}
    for area,path in FILES.items():
        request=urllib.request.Request(f"{ROOT}/{revision}/{path}",headers={"User-Agent":"jaco-litert-review"})
        with urllib.request.urlopen(request,timeout=30) as response:
            raw=response.read(2_000_001)
        if len(raw)>2_000_000:raise ValueError("Source too large")
        result[area]=raw.decode("utf-8")
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--revision",required=True)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--fetch",action="store_true")
    modes.add_argument("--source-dir",type=Path)
    p.add_argument("--output",type=Path,default=Path("bench/litert-evidence.json"))
    a=p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}",a.revision):p.error("Pin a 40-character commit SHA")
    data=retrieve(a.revision) if a.fetch else {k:(a.source_dir/v).read_text(encoding="utf-8") for k,v in FILES.items()}
    result=assess(data,a.revision)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(f"Evidence saved: {a.output}. Native ACFQ integration BLOCKED pending validation.")
if __name__=="__main__":main()
