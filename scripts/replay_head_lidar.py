"""CPU-only moving-head LiDAR replay of logged ELF3 joint and root poses."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from elf3_climbing.paths import OUTPUT

REPO = Path(__file__).resolve().parents[2]
HARDWARE = REPO / "bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation"


def build_scene(robot_path, scene_path):
    """Use the head-equipped robot, the logged robot's joint convention and the replay terrain."""
    robot = ET.parse(robot_path).getroot()
    terrain = ET.parse(scene_path).getroot()
    robot.find("compiler").set("meshdir", str((robot_path.parent / "meshes").resolve()))
    # Only replay kinematics. Load the actual mount relative to its original parent.
    for parent in robot.iter():
        for inc in parent.findall("include"):
            if inc.get("file").endswith("head_lidar_mount.xml"):
                parent.extend(ET.parse(robot_path.parent / inc.get("file")).getroot())
            parent.remove(inc)
    world = robot.find("worldbody")
    for geom in terrain.findall("worldbody/geom"):
        world.append(geom)
    scene = mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))
    return scene


def spherical(azimuth, elevation):
    c = np.cos(elevation)
    return np.column_stack((c * np.cos(azimuth), c * np.sin(azimuth), np.sin(elevation)))


def beam_angles(indices, fov, channels):
    """Equal-solid-angle nonrepeating sampling; physical MID-360 firing tables are unavailable."""
    group, channel = np.divmod(indices, channels)
    h0, h1, v0, v1 = np.deg2rad(fov)
    az = h0 + (h1 - h0) * ((group * (np.sqrt(5) - 1) / 2 + channel / channels) % 1)
    band = (channel + (group * np.sqrt(2)) % 1) / channels
    el = np.arcsin(np.sin(v0) + band * (np.sin(v1) - np.sin(v0)))
    return az, el


def capture(trace, scene_path, robot_path, seed):
    cfg = ET.parse(robot_path.parent / "sensors/head_lidar.xml").getroot()

    def numeric(name):
        elem = cfg.find(f".//numeric[@name='simulation.lidar.head_lidar.{name}']")
        return np.fromstring(elem.get("data"), sep=" ")

    n_beams, hz = map(int, numeric("rate"))
    channels = int(numeric("channels")[0])
    near, far = numeric("range")
    fov = numeric("fov")
    range_sigma, angle_sigma = numeric("noise")[:2]
    angle_sigma = np.deg2rad(angle_sigma)
    interval = 1 / hz
    slices = round(interval / 0.001)
    with np.load(trace, allow_pickle=False) as archive:
        poses = archive["root_pose"].astype("d")
        joints = archive["joint_pos"].astype("d")
        dt, lane = float(archive["dt"]), int(archive["lane"])
        joint_names = archive["joint_names"].astype(str)
    times = (np.arange(len(poses)) + 1) * dt
    rotations = Slerp(times, Rotation.from_quat(poses[:, 3:], scalar_first=True))

    def roots_at(t):
        xyz = np.column_stack([np.interp(t, times, c) for c in poses[:, :3].T])
        return np.column_stack((xyz, rotations(t).as_quat(scalar_first=True)))

    frames = int(np.floor((times[-1] - times[0]) / interval + 1e-9))
    if frames < 1:
        raise ValueError("Trace is shorter than one complete scan")
    ends = times[0] + np.arange(1, frames + 1) * interval
    centers = times[0] + (np.arange(frames * slices) + 0.5) * (interval / slices)
    root = roots_at(centers)
    joint = np.column_stack([np.interp(centers, times, c) for c in joints.T])
    model = build_scene(robot_path, scene_path)
    data = mujoco.MjData(model)
    site = model.site("head_lidar_mount").id
    exclude = cfg.find(".//text[@name='simulation.lidar.head_lidar.exclude_body']").get("data")
    owner = model.body(exclude).id
    joint_addr = [model.jnt_qposadr[model.joint(name).id] for name in joint_names]
    root_addr = model.jnt_qposadr[model.joint("world_joint").id]
    rng = np.random.default_rng(seed)
    n = frames * n_beams
    index = np.arange(n)
    az, el = beam_angles(index, fov, channels)
    rays = spherical(az, el)
    sensed = spherical(az + rng.normal(0, angle_sigma, n), el + rng.normal(0, angle_sigma, n))
    bias = rng.normal(0, range_sigma, n)
    points = np.full((n, 3), np.nan, dtype="f4")
    hit_kind = np.zeros(n, dtype="u1")
    hit_geom = np.full(n, -1, dtype="i4")
    ray_times = np.empty(n)
    origin = np.empty((len(centers), 3))
    orientation = np.empty((len(centers), 3, 3))
    for i, t in enumerate(centers):
        frame, k = divmod(i, slices)
        begin = frame * n_beams + k * n_beams // slices
        end = frame * n_beams + (k + 1) * n_beams // slices
        block = slice(begin, end)
        ray_times[block] = t
        data.qpos[root_addr : root_addr + 7] = root[i]
        data.qpos[joint_addr] = joint[i]
        mujoco.mj_forward(model, data)
        origin[i] = data.site_xpos[site]
        orientation[i] = data.site_xmat[site].reshape(3, 3)
        direction = np.ascontiguousarray(rays[block] @ orientation[i].T)
        distance = np.empty(end - begin, dtype="d")
        geoms = hit_geom[block]
        mujoco.mj_multiRay(
            model,
            data,
            origin[i],
            direction.ravel(),
            None,
            True,
            owner,
            geoms,
            distance,
            None,
            len(distance),
            far,
        )
        intersects = (geoms >= 0) & (distance >= 0) & (distance <= far)
        body = model.geom_bodyid[np.maximum(geoms, 0)]
        self_hit = intersects & (body != 0)
        measured = distance + bias[block]
        valid = intersects & (distance >= near) & (measured >= near) & (measured <= far)
        kind = hit_kind[block]
        kind[valid & ~self_hit] = 1
        # A near robot surface occludes downstream terrain even inside the blind zone.
        kind[self_hit] = 2
        points[block][valid] = (
            origin[i] + (sensed[block][valid] @ orientation[i].T) * measured[valid, None]
        )
    terrain_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == 0]
    box_ids = [i for i in terrain_ids if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX]
    provenance = dict(
        source=str(trace.resolve()),
        lane=lane,
        robot_model=str(robot_path.resolve()),
        scene=str(scene_path.resolve()),
        scan_geometry="Non-repeating equal-solid-angle 4-channel sampling approximation",
        time_model="Complete 10 Hz packets, 1 ms simultaneous ray batches, root SLERP and joint interpolation",
        noise="Independent Gaussian range and angular noise per XML; exact world pose for deskew",
        hit_kind="0 miss/invalid, 1 observed terrain, 2 robot occlusion",
        head_joints="Unlogged head pitch and yaw held at neutral; head_y_link excluded per XML",
        coordinate_frame="Isaac replay environment-local world, root_pose xyz+wxyz at packet delivery",
        seed=seed,
        range_m=[near, far],
        fov_deg=fov.tolist(),
        rays_per_packet=n_beams,
        scan_rate_hz=hz,
    )
    return dict(
        points_w=points,
        scan_index=(index // n_beams).astype("i4"),
        hit_kind=hit_kind,
        hit_geom=hit_geom,
        ray_time_s=ray_times,
        scan_time_s=ends,
        root_pose=roots_at(ends),
        emission_time_s=centers,
        sensor_origin_w=origin,
        sensor_rotation_w=orientation,
        terrain_geomid=np.array(terrain_ids),
        terrain_box_names=[model.geom(i).name for i in box_ids],
        terrain_box_centers_w=model.geom_pos[box_ids],
        terrain_box_half_sizes=model.geom_size[box_ids],
        terrain_box_quat_wxyz=model.geom_quat[box_ids],
        metadata_json=json.dumps(provenance),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model", type=Path, default=OUTPUT / "mount_070_ground_to_steady_rise_20261123/model.xml"
    )
    parser.add_argument("--robot", type=Path, default=HARDWARE / "elf3.xml")
    parser.add_argument("--seed", type=int, default=9500)
    args = parser.parse_args()
    scan = capture(args.trace, args.model, args.robot, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **scan)
    print(
        "Packets:",
        len(scan["scan_time_s"]),
        "miss/terrain/robot:",
        np.bincount(scan["hit_kind"], minlength=3),
    )


if __name__ == "__main__":
    main()
