#!/usr/bin/env python3
"""Read-only LiteRT-LM/Termux runtime compatibility probe.

No model inference, package install, model file scan, process restart, network
except GET localhost /v1/models. Does not print process argv, model paths,
secrets, user data, or files with weights.
"""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

# Official version-pinned source: google-ai-edge/LiteRT-LM/<tag>/WORKSPACE
KNOWN_LITERT_PINS = {
    "0.17.1": "9fe5be45564c868408e6514c8aabb83e211a0911",
    "0.18.0": "26895c9fbcc25c43faa8c1a98cd1fd28951602c3",
}
JACO_PIN = "44455548be35bde21737b8a28ef0d027632c282d"


def installed_versions():
    entries = []
    for name in ("litert-lm", "litert_lm", "ai-edge-litert-lm"):
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        version = dist.version
        label = dist.metadata.get("Name", name)
        if not any(x["name"] == label for x in entries):
            entries.append({"name": label, "version": version})
    return entries


def version_pin(version):
    if not version:
        return {"revision": None, "relationship": "UNKNOWN_VERSION"}
    revision = KNOWN_LITERT_PINS.get(version)
    return {
        "revision": revision,
        "relationship": (
            "DIFFERENT_FROM_JACO_PIN" if revision and revision != JACO_PIN
            else "SAME_AS_JACO_PIN" if revision == JACO_PIN
            else "UNKNOWN_BUILD_PROVENANCE"
        ),
    }


def read_local_models(url):
    # Strict localhost only: must never contact unrelated endpoints.
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("only localhost HTTP /v1/models is permitted")
    if parsed.path != "/v1/models" or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("only /v1/models (no query) is permitted")
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            models = json.load(response)
        return {
            "reachable": True,
            "model_ids": [
                str(item.get("id", ""))[:128]
                for item in models.get("data", [])
                if isinstance(item, dict)
            ][:10],
        }
    except (OSError, TimeoutError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"reachable": False, "error_type": type(exc).__name__}


def command_version(executable):
    if not executable:
        return {"available": False}
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True,
            timeout=5, check=False,
        )
        output = (result.stdout.strip() or result.stderr.strip())
        return {
            "available": True,
            "exit_code": result.returncode,
            "version_output": output[:200],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error_type": type(exc).__name__}


def needed_from_readelf(executable, tool):
    """ELF NEEDED libs only; never execute binary or read model files."""
    if not executable or not tool:
        return []
    try:
        with open(executable, "rb") as f:
            if f.read(4) != b"\x7fELF":
                return []
        p = subprocess.run(
            [tool, "-d", executable], capture_output=True, text=True,
            timeout=5, check=False,
        )
        if p.returncode != 0:
            return []
        return re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", p.stdout)[:24]
    except (OSError, subprocess.TimeoutExpired):
        return []


def executable_kind(cli):
    if not cli:
        return "absent"
    try:
        with open(cli, "rb") as fh:
            return "ELF" if fh.read(4) == b"\\x7fELF" else "launcher_script"
    except OSError:
        return "unreadable"


def probe(url):
    cli = shutil.which("litert-lm")
    dist = installed_versions()
    disk = shutil.disk_usage(str(Path.home()))
    entry = {
        "host": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "termux_present": bool(os.environ.get("PREFIX", "").find("com.termux") >= 0),
            "free_disk_gib": round(disk.free / (1024 ** 3), 2),
        },
        "available_tools": {
            k: bool(shutil.which(k))
            for k in ("python3", "clang", "clang++", "cmake", "ninja",
                      "bazel", "bazelisk", "readelf")
        },
        "cli": {
            "found": bool(cli),
            "kind": executable_kind(cli),
            "version": command_version(cli),
            "needed_shared_libs": needed_from_readelf(cli, shutil.which("readelf")),
        },
        "python_packages": dist,
        "server": read_local_models(url),
        "known_release_revisions": {
            version: version_pin(version) for version in KNOWN_LITERT_PINS
        },
        "jaco_source_revision": JACO_PIN,
        "installed_release_pin": version_pin(dist[0]["version"]) if dist else
                                 version_pin(None),
        "linked_litert_revision_verified": False,
        "interpretation": (
            "A responsive /v1/models endpoint identifies the service, "
            "NOT its linked LiteRT revision or actual CPU/GPU delegate. "
            "Release pin is a source-build expectation, not proof of the "
            "installed wheel's binary provenance."
        ),
    }
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-url", default="http://127.0.0.1:9379/v1/models")
    args = ap.parse_args()
    # Fail fast on non-local URL BEFORE any other probe.
    from urllib.parse import urlsplit
    u = urlsplit(args.models_url)
    if u.scheme != "http" or u.hostname not in ("127.0.0.1", "localhost") \
            or u.path != "/v1/models" or u.query or u.fragment or u.username:
        ap.error("--models-url must be localhost HTTP /v1/models")
    print(json.dumps(probe(args.models_url), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
