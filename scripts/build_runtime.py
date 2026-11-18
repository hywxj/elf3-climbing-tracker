"""Copy the minimum runtime into a new data directory; never modify source assets."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(asset, motion, reference, gripper, output):
    if output.exists():
        raise FileExistsError(f"Choose a new runtime directory: {output}")
    original = json.loads(reference.with_name("manifest.json").read_text())
    if sha(reference) != original["pack_sha256"] or sha(motion) != original["source_sha256"]:
        raise ValueError("Source/reference checksum mismatch")
    # Check the selected model/gripper against the original reference contract.
    selected = {
        "elf3.urdf": asset / "urdf/elf3.urdf",
        "elf3.xml": asset / "xml/elf3.xml",
        "gripper_envelope.json": gripper,
    }
    for old, expected in original["hashes"].items():
        if Path(old).name not in selected or sha(selected[Path(old).name]) != expected:
            raise ValueError(f"Selected asset does not match reference: {old}")
    output.mkdir(parents=True)
    for folder in ("urdf", "xml", "meshes", "usd"):
        shutil.copytree(
            asset / folder,
            output / "assets/elf3_lite" / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "elf3_hiking.usda"),
        )
    (output / "motion").mkdir()
    (output / "reference").mkdir()
    shutil.copy2(motion, output / "motion/source.npz")
    shutil.copy2(reference, output / "reference/reference.npz")
    shutil.copy2(gripper, output / "gripper_envelope.json")
    meta = dict(original)
    meta["source"] = "../motion/source.npz"
    meta["hashes"] = {
        "../assets/elf3_lite/urdf/elf3.urdf": sha(selected["elf3.urdf"]),
        "../assets/elf3_lite/xml/elf3.xml": sha(selected["elf3.xml"]),
        "../gripper_envelope.json": sha(gripper),
    }
    (output / "reference/manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    files = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "bundle_manifest.json").write_text(
        json.dumps({"schema": "elf3_runtime_v1", "files": files}, indent=2)
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "files": len(files),
                "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("asset", "motion", "reference", "gripper", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    build(**{k: v.expanduser().resolve() for k, v in vars(args).items()})
