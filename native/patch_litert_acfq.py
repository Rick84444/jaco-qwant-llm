#!/usr/bin/env python3
"""Opt-in JACO ACFQ shadow integration for an ISOLATED LiteRT source checkout.

Usage:
  python3 native/patch_litert_acfq.py --litert-root /path/to/LiteRT --check
  python3 native/patch_litert_acfq.py --litert-root /path/to/LiteRT --apply

DOES NOT patch installed Termux packages, running libraries, model files,
LiteRT-LM server or JACO services. Compiling patched LiteRT and observing
its callbacks in real Gemma decoding is an independent required step.
"""
import argparse
from pathlib import Path

MARKER="JACO_ACFQ_SHADOW_SOURCE_PATCH_V1"
ENTRY='  tflite::InterpreterOptions interpreter_options;'
INCLUDES='#include "litert/runtime/jaco_acfq_shadow.h"\nextern "C" {\n#include "litert/runtime/jaco_acfq_native/gemma_int4_kernel.c"\n}\n'
HOOK=f"""  // {MARKER}: optional CPU-only Builtin FullyConnected shadow.
#if !defined(LITERT_NO_BUILTIN_OPS)
  const char* jaco_shadow = std::getenv("JACO_ACFQ_SHADOW");
  if (jaco_shadow && jaco_shadow[0]=='1' && jaco_shadow[1]=='\\0' &&
      (hardware_accelerators & kLiteRtHwAcceleratorCpu)) {{
    resolver->AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                         jaco_acfq::ShadowRegistration(), 1, 15);
    std::fprintf(stderr,
        "JACO_ACFQ_SHADOW: registration_installed=1 "
        "native_invocations=0_until_measured\\n");
  }}
#endif
"""
class PatchError(ValueError):
    pass

def apply_patch(source):
    if MARKER in source:
        return source, False
    if source.count(ENTRY)!=1 or source.count('#include "tflite/delegates/xnnpack')>1:
        raise PatchError("LiteRT compiled_model.cc source layout unknown; refusing patch")
    # The patch is inserted before interpreter creation, after resolver setup.
    if 'resolver->AddCustom' not in source or 'ModifyGraphWithDelegate' not in source:
        raise PatchError("not a compatible LiteRT compiled model implementation")
    pos=source.find('#include ')
    if pos<0:raise PatchError("missing include block")
    source=source[:pos]+INCLUDES+source[pos:]
    source=source.replace(ENTRY,HOOK+ENTRY)
    return source,True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--litert-root',required=True,type=Path)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true')
    group.add_argument('--apply',action='store_true')
    args=ap.parse_args()
    root=args.litert_root.resolve()
    dst=root/'litert/runtime/compiled_model.cc'
    if not dst.is_file():
        ap.error(f"not found: {dst}")
    if not (root/'tflite/core/c/common.h').is_file():
        ap.error("expected LiteRT source tree with TFLite headers")
    source=dst.read_text()
    try: patched,changed=apply_patch(source)
    except PatchError as e:ap.error(str(e))
    if args.check:
        print(f"PASS: compatible source; changes_needed={changed}; live_runtime_untouched=1")
        return
    if not changed:
        print("ALREADY PATCHED (no changes)")
        return
    local=Path(__file__).resolve().parent
    targets={
        root/'litert/runtime/jaco_acfq_shadow.h':local/'litert_acfq_shadow.h',
        root/'litert/runtime/jaco_acfq_native/gemma_int4_kernel.h':local/'gemma_int4_kernel.h',
        root/'litert/runtime/jaco_acfq_native/gemma_int4_kernel.c':local/'gemma_int4_kernel.c',
    }
    if any(t.exists() for t in targets):
        ap.error("destination files already present; refusing overwrite")
    backup=dst.with_name('compiled_model.cc.jaco-original')
    if backup.exists():
        ap.error(f"backup already present; refusing overwrite: {backup}")
    backup.write_text(source)
    for output,original in targets.items():
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_bytes(original.read_bytes())
    dst.write_text(patched)
    print("PATCHED isolated LiteRT source; no installed libraries modified.")
    print(f"Backup: {backup}")
    print("Build/link upstream LiteRT independently. Only experimental CPU")
    print("models with JACO_ACFQ_SHADOW=1 may register the wrapper.")
    print("An installed wrapper is NOT proof of real invocation; inspect logs.")

if __name__=="__main__":main()
