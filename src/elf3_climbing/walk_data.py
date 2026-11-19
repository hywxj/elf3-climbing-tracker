"""Read-only source selection; derived walking data belongs in the runtime folder."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

WRIST_NAME = "l_wrist_y_joint"
WRIST_TARGET = -1.30898
# The existing named AMP70 contract: right arm, left arm, waist, right/left leg.
IGNORED_AMP_COLUMNS = (12, 41)


def mask_wrist(features):
    """Same transform for expert and simulated AMP states, without mutating input."""
    if features.shape[-1] != 70:
        raise ValueError("Expected AMP70 features")
    result = features.clone() if hasattr(features, "clone") else features.copy()
    result[..., list(IGNORED_AMP_COLUMNS)] = 0
    return result


def prepare_walk_manifest(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    raw = source.read_bytes()
    manifest = json.loads(raw)
    if manifest["contract"]["amp_joint_names"][12] != WRIST_NAME:
        raise ValueError("Unexpected AMP wrist order")
    # walk_turn is the existing ordinary walking/turning category. Exclude stairs,
    # slopes, stones, stationary-root and treadmill motions from this flat pilot.
    rows = [r for r in manifest["clips"] if r["eligible"] and r["category"] == "walk_turn"]
    if not rows or not any(r["split"] == "train" for r in rows):
        raise ValueError("No eligible walking training clips")
    destination.mkdir(parents=True, exist_ok=False)
    for row in rows:
        relative = Path(row["output"])
        src = (source.parent / relative).resolve()
        if relative.is_absolute() or not src.is_relative_to(source.parent):
            raise ValueError("Motion escaped source manifest directory")
        if hashlib.sha256(src.read_bytes()).hexdigest() != row["output_sha256"]:
            raise ValueError(f"Source hash mismatch: {row['id']}")
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    result = {k: deepcopy(manifest[k]) for k in ("schema_version", "sample_dt", "contract")}
    result["clips"] = rows
    result["selection"] = {
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "category": "walk_turn",
        "train": sum(r["split"] == "train" for r in rows),
        "validation": sum(r["split"] == "validation" for r in rows),
        "physical_execution_validated": False,
    }
    path = destination / "manifest.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


class WristMaskedPrior:
    """Wrap the existing validated sampler; keep original files and split intact."""

    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def sample_pairs(self, count):
        return tuple(mask_wrist(x) for x in self.source.sample_pairs(count))

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            yield self.sample_pairs(mini_batch_size)

    def get_full_frame_batch(self, count):
        return self.sample_pairs(count)[0]
