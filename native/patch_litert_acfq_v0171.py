#!/usr/bin/env python3
"""Source-only adapter for LiteRT-LM 0.17.1's pinned LiteRT dependency.

No installed Termux wheel/library, model data, or running process is changed.
"""
import argparse
from pathlib import Path
import subprocess

from patch_litert_acfq import ENTRY, HOOK, INCLUDES, MARKER, PatchError

PIN = "9fe5be45564c868408e6514c8aabb83e211a0911"
TRACE_MARKER = "JACO_ACFQ_TRACE_0171_V1"
DELEGATE_ANCHOR = "      current_undelegated_nodes = counts.undelegated_nodes;"
END_ANCHOR = "  compiled_model->CheckCpuTensors();"
TRACE = r'''      // JACO_ACFQ_TRACE_0171_V1: aggregate post-delegation evidence only.
      const char* jaco_trace=std::getenv("JACO_ACFQ_SHADOW");
      if (jaco_trace && jaco_trace[0]=='1' && jaco_trace[1]=='\0') {
        std::fprintf(stderr,
          "JACO_ACFQ_SHADOW: delegated_nodes=%d remaining_nodes=%d "
          "cpu_partitions=%d gpu_partitions=%d npu_partitions=%d\n",
          delegated_ops, counts.undelegated_nodes, counts.cpu_partitions,
          counts.gpu_partitions, counts.npu_partitions);
      }
'''
TRACE_FINAL = r'''  // JACO_ACFQ_TRACE_0171_V1: post-graph aggregate; NOT proof of CPU callback.
  const char* jaco_trace_done=std::getenv("JACO_ACFQ_SHADOW");
  if (jaco_trace_done && jaco_trace_done[0]=='1' &&
      jaco_trace_done[1]=='\0') {
    std::fprintf(stderr,
        "JACO_ACFQ_SHADOW: delegation_complete=1 remaining_nodes=%d "
        "cpu_delegated_nodes=%d total_nodes=%d\n",
        current_undelegated_nodes,
        compiled_model->delegation_metrics_.cpu_delegated_node_count,
        compiled_model->delegation_metrics_.total_node_count);
  }
'''


def patch_source(source):
    """Accept only the audited v0.17.1 source anchors; never merge patches."""
    if MARKER in source or TRACE_MARKER in source:
        if MARKER in source and TRACE_MARKER in source:
            return source, False
        raise PatchError("partial or incompatible ACFQ patch present")
    required_once = (
        ENTRY, DELEGATE_ANCHOR, END_ANCHOR,
        "int current_undelegated_nodes =",
    )
    for item in required_once:
        if source.count(item) != 1:
            raise PatchError("unexpected LiteRT v0.17.1 source anchor: " + item)
    if ("      prev_counts = counts;" in source or
            "resolver->AddCustom" not in source or
            "ModifyGraphWithDelegate" not in source):
        raise PatchError("not the expected LiteRT 0.17.1 layout")
    pos = source.find("#include ")
    if pos < 0:
        raise PatchError("no include block")
    changed = source[:pos] + INCLUDES + source[pos:]
    changed = changed.replace(ENTRY, HOOK + ENTRY, 1)
    changed = changed.replace(DELEGATE_ANCHOR,
                              DELEGATE_ANCHOR + "\n" + TRACE, 1)
    changed = changed.replace(END_ANCHOR, TRACE_FINAL + END_ANCHOR, 1)
    return changed, True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--litert-root", required=True, type=Path)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = args.litert_root.expanduser().resolve()
    home = Path.home().resolve()
    if not root.is_relative_to(home) or root.name != "jaco-litert-acfq-0171-source":
        ap.error("expected separate ~/jaco-litert-acfq-0171-source")
    if not (root / ".git").exists():
        ap.error("not a separately cloned LiteRT checkout")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        ap.error("cannot verify pinned Git revision")
    if revision != PIN:
        ap.error("LiteRT SHA mismatch; refusing source patch")
    dst = root / "litert/runtime/compiled_model.cc"
    if not dst.is_file() or not (root / "tflite/core/c/common.h").is_file():
        ap.error("LiteRT source files missing")
    source = dst.read_text()
    try:
        new_source, changes = patch_source(source)
    except PatchError as exc:
        ap.error(str(exc))
    if not changes:
        print("ALREADY PATCHED: 0.17.1 source only")
        return
    try:
        raw = subprocess.run(
            ["git", "-C", str(root), "show", "HEAD:litert/runtime/compiled_model.cc"],
            check=True, capture_output=True
        ).stdout
    except subprocess.CalledProcessError:
        ap.error("cannot retrieve pinned upstream original source")
    if source.encode() != raw:
        ap.error("manual source changes present; refusing overwrite")
    print("PASS: source layout and pinned commit verified")
    if args.check:
        print("CHECK ONLY: no files changed")
        return

    local = Path(__file__).resolve().parent
    targets = {
        root / "litert/runtime/jaco_acfq_shadow.h": local / "litert_acfq_shadow.h",
        root / "litert/runtime/jaco_acfq_native/gemma_int4_kernel.h": local / "gemma_int4_kernel.h",
        root / "litert/runtime/jaco_acfq_native/gemma_int4_kernel.c": local / "gemma_int4_kernel.c",
    }
    backup = dst.with_name("compiled_model.cc.jaco-original")
    if backup.exists() or backup.is_symlink() or any(
        dest.exists() or dest.is_symlink() for dest in targets
    ):
        ap.error("backup/target exists; refusing destructive overwrite")
    if any(not src.is_file() for src in targets.values()):
        ap.error("local ACFQ source files missing")
    try:
        with backup.open("xb") as f:
            f.write(raw)
        for dest, src in targets.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as f:
                f.write(src.read_bytes())
        dst.write_text(new_source)
    except OSError:
        ap.error("partial source preparation; inspect isolated checkout before retry")
    print("PATCHED v0.17.1 source only; no binary compiled or replaced")


if __name__ == "__main__":
    main()
