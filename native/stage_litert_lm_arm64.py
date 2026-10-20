#!/usr/bin/env python3
"""Safely stage, never install, an experimental Android LiteRT-LM .so.

Stage a PRIVATE COPY of the installed Python package under the current
user's home directory. The working litert-lm wheel and :9379 server are
never edited. No process is launched, no model data copied or accessed.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import tempfile

NAME = "jaco-acfq-stage-0171"
DSO = "liblitert-lm-acfq-0171-experimental.so"


def validate_arm64_elf(path):
    with path.open("rb") as f:
        head = f.read(20)
    if (len(head) < 20 or head[:6] != b"\x7fELF\x02\x01" or
            int.from_bytes(head[16:18], "little") != 3 or
            int.from_bytes(head[18:20], "little") != 183):
        raise ValueError("not an Android ARM64 ET_DYN ELF candidate")


def validate_artifact(artifact_dir):
    folder = Path(artifact_dir).expanduser().resolve()
    dso = folder / DSO
    sums = folder / "SHA256SUMS"
    if not dso.is_file() or dso.is_symlink() or not sums.is_file():
        raise ValueError("experimental library or checksum file missing")
    validate_arm64_elf(dso)
    expected = sums.read_text().split(maxsplit=1)[0].lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("invalid SHA256SUMS digest")
    digest = hashlib.sha256()
    with dso.open("rb") as stream:
        while piece := stream.read(1048576):
            digest.update(piece)
    if digest.hexdigest() != expected:
        raise ValueError("artifact checksum mismatch")
    return dso, expected


def installed_package():
    for name in ("litert-lm", "litert-lm-api", "litert-lm-builder"):
        if md.version(name) != "0.17.1":
            raise ValueError(name + " is not v0.17.1")
    package = Path(md.distribution("litert-lm-api").locate_file("litert_lm"))
    if not (package / "_ffi.py").is_file() or not (package / "liblitert-lm.so").is_file():
        raise ValueError("expected installed LiteRT-LM Python package not found")
    return package


def stage_private(package, dso, target):
    package = Path(package).resolve(strict=True)
    dso = Path(dso).resolve(strict=True)
    home = Path.home().resolve()
    target = Path(target).expanduser()
    if target != home / NAME:
        raise ValueError("destination must be exactly ~/{}".format(NAME))
    if target.exists() or target.is_symlink():
        raise FileExistsError("private stage already exists; refusing overwrite")
    if target == package or package.is_relative_to(target):
        raise ValueError("destination overlaps installed wheel")
    if any(p.is_symlink() for p in package.rglob("*")):
        raise ValueError("unexpected symlinks in installed package; refusing copy")
    # All writing remains within a separate temporary directory in $HOME.
    temp = Path(tempfile.mkdtemp(prefix=".jaco-acfq-stage-", dir=home))
    try:
        staging = temp / "litert_lm"
        shutil.copytree(package, staging, symlinks=False)
        shutil.copy2(dso, staging / "liblitert-lm.so")
        marker = {
            "stage": "EXPERIMENTAL_NOT_PRODUCTION",
            "installed_wheel_modified": False,
            "gemma_run_performed": False,
            "use_pythonpath_only": True,
        }
        (temp / "STAGE_MANIFEST.json").write_text(json.dumps(marker, indent=2) + "\n")
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    dso, sha = validate_artifact(args.artifact_dir)
    source = installed_package()
    dest = Path.home().resolve() / NAME
    if dest.exists():
        parser.error("stage already exists; no overwrite permitted")
    if args.check:
        print("CHECK ONLY: ARM64 candidate SHA256=" + sha)
        print("CHECK ONLY: installed wheel untouched; no stage created")
        return
    stage_private(source, dso, dest)
    print("STAGED isolated Python package: " + str(dest))
    print("No process launched; the installed wheel and Gemma remain unchanged")
    print("Do NOT point the production service at this experimental library")


if __name__ == "__main__":
    main()
