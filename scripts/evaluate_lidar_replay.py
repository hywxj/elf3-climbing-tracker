"""Score a causal head-LiDAR map against the platform on a recorded physical rollout."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from elf3_climbing.contact_edge import edge_from_scan, sample_height_scan, scan_axes
from elf3_climbing.lidar_map import elevation_from_world
from elf3_climbing.paths import OUTPUT


def yaw_axes(q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scans", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=OUTPUT
        / "platform_contacts_20261120_070/isaac_v4_070_surface_reference/reference.npz",
    )
    parser.add_argument("--history-seconds", type=float, default=0.30)
    args = parser.parse_args()
    if args.history_seconds <= 0:
        parser.error("history-seconds must be positive")
    torch.set_num_threads(1)
    with np.load(args.reference) as ref:
        centers, halves, quats = (
            ref[k]
            for k in ("terrain_box_centers_w", "terrain_box_half_sizes", "terrain_box_quat_wxyz")
        )
    with np.load(args.trace) as trace:
        start = trace["root_pose"][0, :2]
        times = (np.arange(len(trace["root_pose"])) + 1) * float(trace["dt"])
        measures = dict(zip(trace["metric_names"], trace["metrics"].T, strict=True))
        first_brace = times[np.flatnonzero(measures["bilateral_braced"])[0]]
        first_hold = times[np.flatnonzero(measures["knee_hold_s"] > 0)[0]]
    with np.load(args.scans) as cloud:
        times_s, ids, pose = cloud["scan_time_s"], cloud["scan_index"], cloud["root_pose"]
        points, kind, ray_time = cloud["points_w"], cloud["hit_kind"], cloud["ray_time_s"]
    if len(times_s) == 0 or not np.all(np.diff(times_s) > 0):
        raise ValueError("Expected chronologically ordered completed scans")
    if not np.all(ray_time <= times_s[ids] + 1e-6):
        raise ValueError("A scan contains a future ray")
    ground_truth, _ = sample_height_scan(
        *[
            torch.as_tensor(a, dtype=torch.float32)
            for a in (pose[:, :3], pose[:, 3:], centers, halves, quats)
        ]
    )
    gx, gy = scan_axes(ground_truth)
    xy = np.stack(np.meshgrid(gx.numpy(), gy.numpy(), indexing="ij"), axis=-1)
    upper = np.argmax(centers[:, 2] + halves[:, 2])
    center, half = centers[upper, :2], halves[upper, :2]
    axes = yaw_axes(quats[upper])
    initial_local = (start - center) @ axes
    face = int(np.argmax(np.abs(initial_local) / half))
    toward_start = np.sign(initial_local[face]) * axes[:, face]
    front = center + toward_start * half[face]
    inward, tangent = -toward_start, axes[:, 1 - face]
    rows = []
    for i, t in enumerate(times_s):
        # Only completed packets and returns inside the history window are available.
        use = (ids <= i) & (ray_time > t - args.history_seconds) & (kind == 1)
        scan = elevation_from_world(points[use], pose[i])
        h = torch.as_tensor(scan.heights, dtype=torch.float32)[None]
        v = torch.as_tensor(scan.valid, dtype=torch.bool)[None]
        edge = edge_from_scan(h, v, version=2)[0].numpy()
        rotation = yaw_axes(pose[i, 3:])
        world = xy @ rotation.T + pose[i, :2]
        side = (world - front) @ inward
        in_band = (abs(side) <= 0.15) & (abs((world - front) @ tangent) < half[1 - face] - 0.05)
        valid = v[0].numpy()
        distance = float((front - pose[i, :2]) @ inward)
        declared = bool(edge[5] > 0)
        angle = (
            float(np.degrees(np.arccos(np.clip((rotation @ edge[1:3]) @ inward, -1, 1))))
            if declared
            else None
        )
        dist_error = abs(float(edge[0]) - distance) if declared else None
        measured = valid & in_band
        errors = np.abs(np.asarray(scan.heights)[measured] - ground_truth[i].numpy()[measured])
        frame = ids == i
        rows.append(
            dict(
                time_s=round(float(t), 4),
                phase="approach"
                if t < first_brace
                else "braced"
                if t < first_hold
                else "knee_hold",
                rays=int(frame.sum()),
                self_hits=int((frame & (kind == 2)).sum()),
                valid_cells=int(valid.sum()),
                band_cells=int(in_band.sum()),
                band_seen=int(measured.sum()),
                band_height_abs_error_sum_m=float(errors.sum()),
                detected_edge=declared,
                distance_error_m=dist_error,
                normal_error_deg=angle,
                matched_front=bool(declared and angle <= 15 and dist_error <= 0.10),
            )
        )

    def summarize(part):
        band = sum(r["band_cells"] for r in part)
        seen = sum(r["band_seen"] for r in part)
        rays = sum(r["rays"] for r in part)
        return dict(
            frames=len(part),
            frames_with_detected_edge=sum(r["detected_edge"] for r in part),
            frames_matching_front=sum(r["matched_front"] for r in part),
            band_seen_fraction=seen / band if band else None,
            band_observed_height_mae_m=sum(r["band_height_abs_error_sum_m"] for r in part) / seen
            if seen
            else None,
            self_hit_fraction=sum(r["self_hits"] for r in part) / rays if rays else None,
        )

    report = dict(
        scans=str(args.scans),
        trace=str(args.trace),
        reference=str(args.reference),
        history_seconds=args.history_seconds,
        first_brace_s=float(first_brace),
        first_hold_s=float(first_hold),
        match_thresholds={"distance_m": 0.10, "normal_deg": 15},
        overall=summarize(rows),
        phases={
            p: summarize([r for r in rows if r["phase"] == p])
            for p in ("approach", "braced", "knee_hold")
        },
        rows=rows,
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["overall"]))


if __name__ == "__main__":
    main()
