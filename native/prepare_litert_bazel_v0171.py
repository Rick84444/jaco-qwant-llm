#!/usr/bin/env python3
"""Prepare the pinned LiteRT 0.17.1 *Android Bazel* source layout.

Only mutates an isolated ~/jaco-litert-acfq-0171-source checkout already
prepared with patch_litert_acfq_v0171.py. Never touches installed LiteRT-LM.
Linux/CMake retains its separate source adapter path unchanged.
"""
import argparse
from pathlib import Path
import subprocess

from patch_litert_acfq import INCLUDES, MARKER, PatchError
from patch_litert_acfq_v0171 import PIN, TRACE_MARKER

# This C file is included inline by the CMake prototype. Bazel must instead
# compile it as an independent C source. Otherwise source inclusion plus a
# Bazel srcs entry would link duplicate native functions.
COMPILED_INCLUDES = INCLUDES
LINKED_INCLUDES = '#include "litert/runtime/jaco_acfq_shadow.h"\n'
BAZEL_RULE_ANCHOR = '''cc_library(
    name = "compiled_model",
    srcs = [
        "compiled_model.cc",
    ],
    hdrs = [
        "compiled_model.h",
        "stub_op_resolver.h",
    ],
'''
BAZEL_RULE_REPLACEMENT = '''cc_library(
    name = "compiled_model",
    srcs = [
        "compiled_model.cc",
        "jaco_acfq_native/gemma_int4_kernel.c",
    ],
    hdrs = [
        "compiled_model.h",
        "stub_op_resolver.h",
        "jaco_acfq_shadow.h",
        "jaco_acfq_native/gemma_int4_kernel.h",
    ],
'''


def prepare_bazel(source, build):
    """Pure fail-closed transformer: source and BUILD are one logical patch."""
    if source.count(MARKER) != 1 or source.count(TRACE_MARKER) != 2:
        raise PatchError("pinned LiteRT v0.17.1 ACFQ source patch absent/incompatible")
    if source.count(COMPILED_INCLUDES) != 1:
        raise PatchError("expected inline native C inclusion exactly once")
    if build.count(BAZEL_RULE_ANCHOR) != 1:
        raise PatchError("pinned LiteRT compiled_model Bazel rule changed")
    if "jaco_acfq_" in build:
        raise PatchError("BUILD already contains a JACO ACFQ declaration")
    return (source.replace(COMPILED_INCLUDES, LINKED_INCLUDES, 1),
            build.replace(BAZEL_RULE_ANCHOR, BAZEL_RULE_REPLACEMENT, 1))


def pinned_original(root, repo_path):
    completed = subprocess.run(
        ["git", "-C", str(root), "show", "HEAD:" + repo_path],
        capture_output=True, check=True)
    return completed.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--litert-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.litert_root.expanduser().resolve()
    if root != (Path.home().resolve() / "jaco-litert-acfq-0171-source"):
        parser.error("only the separate ~/jaco-litert-acfq-0171-source is allowed")
    if not (root / ".git").is_dir():
        parser.error("not an isolated LiteRT Git checkout")
    try:
        actual_pin = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        parser.error("unable to verify LiteRT Git SHA")
    if actual_pin != PIN:
        parser.error("LiteRT revision mismatch, refusing Bazel patch")

    source_file = root / "litert/runtime/compiled_model.cc"
    build_file = root / "litert/runtime/BUILD"
    original_source_backup = source_file.with_name("compiled_model.cc.jaco-original")
    build_backup = build_file.with_name("BUILD.jaco-original")
    for path in (source_file, build_file, original_source_backup,
                 root / "litert/runtime/jaco_acfq_shadow.h",
                 root / "litert/runtime/jaco_acfq_native/gemma_int4_kernel.h",
                 root / "litert/runtime/jaco_acfq_native/gemma_int4_kernel.c"):
        if not path.is_file():
            parser.error("expected pinned prepared source missing: " + str(path))
    if build_backup.exists() or build_backup.is_symlink():
        parser.error("BUILD backup already present; refusing repeated patch")
    try:
        original_cc = pinned_original(root, "litert/runtime/compiled_model.cc")
        original_build = pinned_original(root, "litert/runtime/BUILD")
    except subprocess.CalledProcessError:
        parser.error("cannot verify pristine upstream BUILD and source")
    if original_source_backup.read_bytes() != original_cc:
        parser.error("original source backup differs from upstream pinned source")
    source_text = source_file.read_text()
    build_text = build_file.read_text()
    if build_text.encode() != original_build:
        parser.error("unexpected changes to upstream Bazel BUILD")
    try:
        updated_source, updated_build = prepare_bazel(source_text, build_text)
    except PatchError as error:
        parser.error(str(error))
    if args.check:
        print("CHECK ONLY: pinned ACFQ Bazel rule and C linkage ready; no mutations")
        return
    try:
        with build_backup.open("xb") as backup:
            backup.write(original_build)
        build_file.write_text(updated_build)
        source_file.write_text(updated_source)
    except OSError as error:
        parser.error("partial isolated Bazel patch, inspect source checkout: " + str(error))
    print("PATCHED isolated Android Bazel graph: C source compiled once; 2 headers declared")


if __name__ == "__main__":
    main()
