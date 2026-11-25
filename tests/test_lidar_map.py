import numpy as np
import torch

from elf3_climbing.contact_edge import edge_from_scan
from elf3_climbing.contact_point_actor import MISSING_HEIGHT, PointHistory, point_sensor
from elf3_climbing.lidar_map import elevation_from_world


def test_map_uses_control_pose_yaw_and_samples_one_surface_for_both_grids():
    # Combined 90-degree world yaw and 60-degree roll. Translation and yaw should be
    # removed, but the z-axis of the elevation map always stays parallel to gravity.
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    pose = np.r_[[2.0, 3.0, 1.20], np.array([c, s, s, c]) / np.sqrt(2)]
    cloud = np.array(
        [
            [1.80, 3.20, 0.01],  # x=.20 ahead, y=.20 left
            [1.80, 3.20, 0.71],
            [2.60, 2.20, 0.12],  # x=-.80, y=-.60; only the base-grid window includes it
            [2.00, 4.41, 0.90],  # x=1.41; nearest centre at 1.40 is in the dense window only
            [np.nan, 0, 3],
            [90, 90, 90],
        ]
    )
    grid = elevation_from_world(cloud, pose)
    assert grid.valid.shape == (39, 37)
    assert grid.valid.sum() == 2
    np.testing.assert_allclose(grid.heights[14, 22], -0.49)
    base_h, base_v = grid.terrain.reshape(2, 9, 7)
    np.testing.assert_allclose(base_h[4, 4], grid.heights[14, 22])
    np.testing.assert_allclose(base_h[0, 0], -1.08)
    np.testing.assert_allclose(grid.heights[38, 18], -0.30)
    assert base_v.sum() == 2 and base_v[4, 4] and base_v[0, 0]
    assert (grid.heights[~grid.valid] == 0).all()


def test_policy_replaces_the_coarse_grid_edge_and_reachable_cells_from_lidar():
    x, y = np.meshgrid(np.arange(-16, 29) / 20, np.arange(-18, 19) / 20, indexing="ij")
    z = np.where(x >= 0.45, 0.70, 0.0)
    point_cloud = np.stack((x, y, z), -1).reshape(-1, 3)
    pose = [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]
    measured = elevation_from_world(point_cloud, pose)
    empty = elevation_from_world(np.empty((0, 3)), pose)
    lidar = [
        np.stack((getattr(measured, k), getattr(empty, k))) for k in ("heights", "valid", "terrain")
    ]
    raw = {
        "root_pos": torch.tensor([pose[:3], pose[:3]]),
        "root_quat": torch.tensor([pose[3:], pose[3:]]),
        "site_pos": torch.zeros(2, 8, 3),
    }
    base = torch.zeros(2, 254)
    base[:, 96:222] = 42  # conspicuous mock truth terrain must never reach the measured path
    base[:, 246:254] = 3
    core, candidates = point_sensor(base, raw, None, None, None, lidar=lidar)
    torch.testing.assert_close(core[:, 96:222], torch.from_numpy(lidar[2]))
    torch.testing.assert_close(core[:, 246:254], torch.full((2, 8), 3.0))
    expected_edge = edge_from_scan(torch.from_numpy(lidar[0]), torch.from_numpy(lidar[1]))
    torch.testing.assert_close(core[:, 254:], expected_edge)
    assert core[0, 259] == 1 and core[1, 259] == 0
    assert abs(core[0, 254].item() - 0.425) < 0.005
    assert candidates.shape == (2, 380)
    torch.testing.assert_close(candidates[0], torch.from_numpy(measured.heights[::2, ::2].ravel()))
    assert (candidates[1] == MISSING_HEIGHT).all()
    history = PointHistory(2, "cpu")
    obs = history.encode(base, raw, None, None, None, 0, lidar=lidar)
    torch.testing.assert_close(obs[:, :262], core)
    torch.testing.assert_close(obs[:, -380:], candidates)
