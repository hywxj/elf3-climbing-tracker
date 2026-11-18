"""Offline checks for a transferred runtime and the pinned local PPO implementation."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    from elf3_climbing.paths import RUNTIME, DEFAULT_PACK
    from elf3_climbing.reference import read_pack
    import torch
    import rsl_rl

    manifest = json.loads((RUNTIME / "bundle_manifest.json").read_text())
    if manifest["schema"] != "elf3_runtime_v1":
        raise ValueError("Unknown runtime schema")
    for name, expected in manifest["files"].items():
        path = (RUNTIME / name).resolve()
        if not path.is_relative_to(RUNTIME) or sha(path) != expected:
            raise ValueError(f"Runtime checksum mismatch: {name}")
    data, meta = read_pack(DEFAULT_PACK)
    repo = Path(__file__).resolve().parents[1]
    pinned = json.loads((repo / "DEPENDENCIES.json").read_text())
    rsl_root = Path(rsl_rl.__file__).resolve().parent
    for name, expected in pinned["rsl_rl_sha256"].items():
        if sha(rsl_root / name) != expected:
            raise ValueError(f"PPO dependency mismatch: {name}; install third_party/rsl_rl")
    versions = {
        key: importlib.metadata.version(key)
        for key in ("isaaclab", "isaacsim", "torch", "mujoco", "scipy")
    }
    mismatches = {
        key: [value, versions[key]]
        for key, value in pinned["versions"].items()
        if key in versions and versions[key] != value
    }
    if mismatches:
        raise ValueError(f"Environment version mismatch [expected, actual]: {mismatches}")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "files/dependencies only, NOT dynamics or training",
                "runtime": str(RUNTIME),
                "verified_files": len(manifest["files"]),
                "reference_frames": len(data["timestamps"]),
                "full_motion_training_ready": meta["full_motion_training_ready"],
                "versions": versions,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
