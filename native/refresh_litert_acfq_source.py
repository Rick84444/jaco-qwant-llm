#!/usr/bin/env python3
"""Refresh the first-generation JACO hook in an isolated sparse LiteRT tree.

Checks exact source revision and untouched original backup; only then replaces
recognized earlier source/header with current reviewed versions. Does not
restart or modify the installed LiteRT-LM server, packages, or model data.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from patch_litert_acfq import ENTRY, HOOK, INCLUDES, apply_patch

PIN = "44455548be35bde21737b8a28ef0d027632c282d"
KNOWN_HEADERS = ("6ef3c0b", "4dca4ef")


def expected_legacy_source(original):
    """Reproduce the earlier no-delegate-trace patch without any guesswork."""
    if original.count(ENTRY) != 1 or "#include " not in original:
        raise ValueError("original LiteRT source has unexpected anchors")
    start = original.index("#include ")
    old = original[:start] + INCLUDES + original[start:]
    return old.replace(ENTRY, HOOK + ENTRY)


def classify_source(original, existing):
    """Only accept exact former or current generated code; refuse manual edits."""
    refreshed, changed = apply_patch(original)
    if not changed:
        raise ValueError("original source contains a JACO marker unexpectedly")
    if existing == refreshed:
        return refreshed, "already-current"
    if existing == expected_legacy_source(original):
        return refreshed, "verified-legacy"
    raise ValueError(
        "unexpected edits in isolated compiled_model.cc; refusing overwrite"
    )


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True
    ).stdout


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, content):
    fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix=".jaco-refresh-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(temp, path.stat().st_mode & 0o777)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def plan(source_dir):
    root = source_dir.expanduser().resolve()
    home = Path.home().resolve()
    if not root.is_relative_to(home) or root == home or (
        root.name != "jaco-litert-acfq-source"
    ):
        raise ValueError("source must be ~/jaco-litert-acfq-source")
    if not (root / ".git").exists():
        raise ValueError("expected a separate upstream Git checkout")
    if git(root, "rev-parse", "HEAD").decode().strip() != PIN:
        raise ValueError("upstream HEAD differs from pinned LiteRT revision")

    compiled = root / "litert/runtime/compiled_model.cc"
    original_path = compiled.with_name("compiled_model.cc.jaco-original")
    header_path = root / "litert/runtime/jaco_acfq_shadow.h"
    repo = Path(__file__).resolve().parent.parent
    for path in (compiled, original_path, header_path):
        if not path.is_file():
            raise ValueError("missing expected isolated file: " + str(path))

    original = original_path.read_bytes()
    upstream = git(root, "show", "HEAD:litert/runtime/compiled_model.cc")
    if upstream != original:
        raise ValueError("source backup differs from pinned upstream Git blob")
    old_src = compiled.read_bytes()
    fresh, state = classify_source(
        original.decode("utf-8"), old_src.decode("utf-8")
    )
    fresh_src = fresh.encode("utf-8")

    new_header = (repo / "native/litert_acfq_shadow.h").read_bytes()
    old_header = header_path.read_bytes()
    known_headers = {new_header}
    for commit in KNOWN_HEADERS:
        try:
            known_headers.add(git(repo, "show", f"{commit}:native/litert_acfq_shadow.h"))
        except subprocess.CalledProcessError:
            continue
    if old_header not in known_headers:
        raise ValueError("unknown isolated shadow header; refusing overwrite")

    changes = []
    if old_src != fresh_src:
        changes.append((compiled, old_src, fresh_src))
    if old_header != new_header:
        changes.append((header_path, old_header, new_header))
    return state, changes


def main():
    ap = argparse.ArgumentParser()
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    ap.add_argument("--source-dir", type=Path,
                    default=Path.home() / "jaco-litert-acfq-source")
    args = ap.parse_args()
    try:
        state, changes = plan(args.source_dir)
        print(f"ISOLATED source state={state}; changes={len(changes)}")
        for path, before, after in changes:
            print(f"  {path}: {sha(before)[:12]} -> {sha(after)[:12]}")
        if args.check:
            print("CHECK ONLY: no source or running model modified")
            return 0
        if not changes:
            print("ALREADY CURRENT; no modifications")
            return 0
        for path, before, after in changes:
            backup = path.with_name(path.name + ".jaco-before-refresh")
            if backup.exists():
                raise ValueError("refresh backup already exists: " + str(backup))
        # Save both older versions before touching either original file.
        for path, before, after in changes:
            backup = path.with_name(path.name + ".jaco-before-refresh")
            with backup.open("xb") as fh:
                fh.write(before)
                fh.flush()
                os.fsync(fh.fileno())
        for path, before, after in changes:
            atomic_write(path, after)
        print("REFRESH PASS: isolated source upgraded, old files backed up")
        print("No binaries built; Gemma on port 9379 is unaffected")
        return 0
    except (ValueError, UnicodeError, OSError,
            subprocess.CalledProcessError) as exc:
        ap.exit(2, f"REFRESH REFUSED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
