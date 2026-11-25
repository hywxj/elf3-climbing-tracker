"""Project measured terrain returns into the contact policy's gravity-aligned terrain grids."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LidarElevation:
    heights: np.ndarray  # 39 x 37, metres relative to the current root; zero in unknown cells.
    valid: np.ndarray  # 39 x 37, measured cells.
    terrain: np.ndarray  # 63 base-grid heights followed by 63 measured flags, x-major.


def elevation_from_world(points_w, root_pose_wxyz):
    """Map world-frame returns to the policy at the supplied control-time pose (xyz, wxyz).

    The caller supplies terrain returns available at the control time, transformed using the
    head pose at each emission. This function can reproject earlier returns using the new root
    pose. Each 5 cm cell stores its highest measured return, including hits on vertical faces.
    """
    points = np.asarray(points_w)
    pose = np.asarray(root_pose_wxyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected measured world points of shape (n, 3)")
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite root pose (x, y, z, w, x, y, z)")
    qnorm = np.linalg.norm(pose[3:])
    if qnorm < 1e-9:
        raise ValueError("Invalid root quaternion")
    w, x, y, z = pose[3:] / qnorm
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    relative = points[np.isfinite(points).all(axis=1)] - pose[:3]
    dx, dy, dz = relative.T
    forward = dx * np.cos(yaw) + dy * np.sin(yaw)
    left = dy * np.cos(yaw) - dx * np.sin(yaw)

    # Union of the original grids. Cell centres at x=-0.80..1.40 and y=-0.90..0.90.
    ix = np.floor((forward + 0.80) / 0.05 + 0.5).astype(np.int64)
    iy = np.floor((left + 0.90) / 0.05 + 0.5).astype(np.int64)
    inside = (ix >= 0) & (ix < 45) & (iy >= 0) & (iy < 37)
    surface = np.full((45, 37), -np.inf, dtype=np.float32)
    np.maximum.at(surface, (ix[inside], iy[inside]), dz[inside])
    measured = np.isfinite(surface)
    surface[~measured] = 0

    # Match TrackerEnv.grid: 9 x-coordinates (-0.80..1.20) and 7 y-coordinates (-0.60..0.60).
    base = surface[:41:5, 6:31:4]
    base_valid = measured[:41:5, 6:31:4]
    terrain = np.concatenate((base.clip(-3, 3).ravel(), base_valid.ravel())).astype(np.float32)
    return LidarElevation(surface[6:], measured[6:], terrain)
