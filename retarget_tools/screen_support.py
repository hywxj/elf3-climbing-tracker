"""Screen elbow-to-wrist coverage of a platform; never infer support forces.

The source bone centreline is only a coarse geometric rejection test. Even a
candidate needs mesh-surface and contact validation before retargeting.
"""

import numpy as np


def bouts(mask, fps):
    values = np.pad(np.asarray(mask, dtype=bool), (1, 1))
    starts = np.flatnonzero(~values[:-1] & values[1:])
    ends = np.flatnonzero(values[:-1] & ~values[1:])
    return [[float(a / fps), float(b / fps)] for a, b in zip(starts, ends)]


def screen(elbows, wrists, fps, boxes):
    """Inputs are (frames, left/right, xyz), in the same world as the boxes."""
    elbows = np.asarray(elbows)
    wrists = np.asarray(wrists)
    if elbows.shape != wrists.shape or elbows.ndim != 3 or elbows.shape[1:] != (2, 3):
        raise ValueError("Expected matching (frames,2,3) forearm endpoints")
    fraction = np.linspace(0, 1, 11)
    points = (
        elbows[:, :, None, :] + (wrists - elbows)[:, :, None, :] * fraction[None, None, :, None]
    )
    mid = (elbows + wrists) / 2
    speed = np.linalg.norm(np.gradient(mid, axis=0) * fps, axis=-1)
    dual = np.zeros(len(elbows), bool)
    stationary_dual = dual.copy()
    any_side = np.zeros((len(elbows), 2), bool)
    elbow_near = [[], []]
    max_coverage = np.zeros(2)
    best_tilt = np.full(2, np.inf)
    details = []
    for i, box in enumerate(boxes):
        c, R, size = box["center"], box["rotation"], box["half_size"]
        if np.ndim(c) == 2:
            high = c[:, 2] + R[:, 2, 2] * size[2] > 0.15
            if not high.any():
                continue
            local = np.einsum("tsvi,tij->tsvj", points - c[:, None, None, :], R)
            direction = np.einsum("tsi,tij->tsj", wrists - elbows, R)
        else:
            if c[2] + R[2, 2] * size[2] < 0.15:
                continue
            high = np.ones(len(elbows), bool)
            local = (points - c) @ R
            direction = (wrists - elbows) @ R
        heights = local[:, :, :, 2] - size[2]
        inside = (np.abs(local[:, :, :, :2]) <= size[:2]).all(-1)
        # Deliberately broad, unscaled 8cm band. Absence even in this band
        # rejects forearm support; presence does not establish contact.
        near = inside & (heights >= -0.01) & (heights <= 0.08)
        coverage = near.mean(-1)
        tilt = np.degrees(
            np.arctan2(np.abs(direction[:, :, 2]), np.linalg.norm(direction[:, :, :2], axis=-1))
        )
        valid = (coverage >= 0.8) & (tilt <= 20) & high[:, None]
        any_side |= valid
        both = valid.all(1)
        dual |= both
        stationary_dual |= both & (speed <= 0.15).all(1)
        hand_near = inside[:, :, -1] & (np.abs(heights[:, :, -1]) < 0.08)
        for k in range(2):
            values = heights[hand_near[:, k], k, 0]
            if len(values):
                elbow_near[k].extend(values.tolist())
            max_coverage[k] = max(max_coverage[k], float(coverage[:, k].max()))
            if hand_near[:, k].any():
                best_tilt[k] = min(best_tilt[k], float(tilt[hand_near[:, k], k].min()))
        spans = bouts(both, fps)
        details.append(
            {
                "box": i,
                "dual_forearm_bouts_s": spans,
                "longest_dual_forearm_bout_s": max([b - a for a, b in spans] or [0.0]),
            }
        )
    spans = bouts(dual, fps)
    longest = max([b - a for a, b in spans] or [0.0])
    return {
        "status": "CANDIDATE" if longest >= 0.2 else "FAIL_GEOMETRIC_SCREEN",
        "geometric_candidate": bool(longest >= 0.2),
        "scope": "Coarse elbow-to-wrist centreline geometry only; not actual surface contact or load bearing",
        "thresholds": {
            "minimum_forearm_fraction_over_top": 0.8,
            "centreline_height_band_m": [-0.01, 0.08],
            "max_forearm_tilt_deg": 20,
            "stationary_midpoint_speed_m_s": 0.15,
            "minimum_dual_bout_s": 0.2,
        },
        "longest_dual_forearm_bout_s": float(longest),
        "dual_forearm_bouts_s": spans,
        "longest_low_speed_dual_forearm_bout_s": max(
            [b - a for a, b in bouts(stationary_dual, fps)] or [0.0]
        ),
        "longest_each_forearm_bout_s": [
            max([b - a for a, b in bouts(any_side[:, k], fps)] or [0.0]) for k in range(2)
        ],
        "max_forearm_fraction_near_top": max_coverage.tolist(),
        "elbow_height_when_wrist_near_top_min_m": [min(v) if v else None for v in elbow_near],
        "min_forearm_tilt_when_wrist_near_top_deg": [
            float(v) if np.isfinite(v) else None for v in best_tilt
        ],
        "platforms": details,
        "force_status": "NOT RUN",
    }


def inspect_bvh(path, stride=4):
    """Use the recorded prop transforms, not a static average of moving props."""
    from bvh import load_bvh
    from pathlib import Path
    import hashlib

    a = load_bvh(path, stride=stride)
    names = a["names"]
    p = a["positions"]
    world = names.index("Z_UP_WORLD_ORIGIN")
    boxes = []
    props = []
    for node in a["bones"]:
        if node["parent"] != world or node["name"] == "Hips":
            continue
        ids = [
            i
            for i, n in enumerate(names)
            if n.startswith(node["name"] + "_") and n.endswith("__end")
        ]
        if not ids:
            continue
        corners = p[:, ids]
        center = corners.mean(1)
        R = a["rotations"][:, names.index(node["name"])].copy()
        flip = R[:, 2, 2] < 0
        R[flip, :, 0] *= -1
        R[flip, :, 2] *= -1
        local = np.einsum("tvi,tij->tvj", corners - center[:, None], R)
        half = np.median((local.max(1) - local.min(1)) / 2, axis=0)
        boxes.append({"center": center, "rotation": R, "half_size": half})
        props.append(
            {
                "name": node["name"],
                "half_size_m": half.tolist(),
                "center_peak_to_peak_m": np.ptp(center, axis=0).tolist(),
            }
        )
    result = screen(
        p[:, [names.index(s + "ForeArm") for s in ("Left", "Right")]],
        p[:, [names.index(s + "Hand") for s in ("Left", "Right")]],
        a["fps"],
        boxes,
    )
    R = a["rotations"][:, names.index("Spine2")]
    tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1, 1)))
    result.update(
        file=str(Path(path)),
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        frames=len(p),
        fps=a["fps"],
        props=props,
        source_first_torso_inclination_deg=float(tilt[0]),
        source_first_second_inclination_range_deg=[
            float(tilt[: round(a["fps"])].min()),
            float(tilt[: round(a["fps"])].max()),
        ],
    )
    return result


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_bvh(args.bvh), indent=2, ensure_ascii=False))
