"""Official ELF3, elbow-supported ascent. Stage-one contact commands come from the user's v4."""

import json
from pathlib import Path

import gymnasium as gym
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from .reference import digest

SITES = (
    "left_forearm",
    "right_forearm",
    "right_knee",
    "left_knee",
    "right_top_foot",
    "left_top_foot",
)
PHASES = ("preload", "push", "right_knee", "left_knee", "right_foot", "left_foot", "stand")
# A site may remain as a support while another site acquires its target.
MASKS = np.array(
    [
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
    ],
    dtype=bool,
)
DEFAULT_MOTION = (
    Path.home()
    / "humanoid_datasheet/datasets/elf3_climbing/elf3/elf3_full_elbow_climb_demo_v4_candidate.npz"
)


class AscentPrior:
    def __init__(self, path, model):
        self.path = Path(path)
        self.sha = digest(path)
        report = json.loads(self.path.with_suffix(".checks.json").read_text())
        if report["summary"]["candidate_sha256"] != self.sha:
            raise ValueError("Motion/checks hash mismatch")
        with np.load(path, allow_pickle=False) as a:
            self.time = a["timestamps"]
            self.joints = a["joint_pos"]
            self.motor = np.array([model.actuator(str(n)).id for n in a["joint_names"]])
            self.start = int(np.flatnonzero(a["takeoff_stage"] == "PRELOAD")[0]) - 1
            mask = a["geometric_contact_mask"].astype(bool)
            names = a["contact_region_names"].tolist()
            center = a["terrain_box_centers_w"][0]
            axes = Rotation.from_quat(a["terrain_box_quat_wxyz"][0], scalar_first=True).as_matrix()
            half = a["terrain_box_half_sizes"][0]
            approach = (a["base_pos_w"][0] - center) @ axes
            axis = np.argmax(abs(approach[:2]) / half[:2])
            forward = -np.sign(approach[axis]) * axes[:, axis]
            left = np.cross([0, 0, 1], forward)
            self.root_delta = (a["base_pos_w"] - a["base_pos_w"][self.start]) @ np.column_stack(
                (forward, left, [0, 0, 1])
            )
            rotations = Rotation.from_quat(a["base_quat_wxyz"], scalar_first=True)
            self.rotation_delta = rotations[self.start].inv() * rotations
            bottom = int(np.flatnonzero(a["takeoff_stage"] == "PUSH_OFF")[0]) - 1
            self.preload_delta = (a["pelvis_w"][bottom] - a["pelvis_w"][self.start]) @ np.column_stack((forward, left, [0, 0, 1]))
            self.drop = float(report["assisted_takeoff"]["summary"]["preload_pelvis_drop_m"])
            self.frame = [
                self.start,
                np.flatnonzero(a["takeoff_stage"] == "PUSH_OFF")[0],
                np.flatnonzero(a["takeoff_stage"] == "ASSISTED_KNEE_TUCK")[0],
            ]
            anchors = []
            for site in SITES:
                times = np.flatnonzero(mask[self.start :, names.index(site)]) + self.start
                frame = next(
                    int(i)
                    for i in times
                    if any(e["region"] == site for e in report["frames"][i]["events"])
                )
                points = [
                    e["point_w"] for e in report["frames"][frame]["events"] if e["region"] == site
                ]
                point = np.mean(points, axis=0) - center
                anchors.append([point @ forward + half[axis], point @ left])
                if site.endswith("knee") or site.endswith("foot"):
                    self.frame.append(frame)
            self.xy = np.asarray(anchors)
            if len(self.frame) != len(PHASES) or np.any(np.diff(self.frame) <= 0):
                raise ValueError(
                    "Expected v4 double push-off, right knee first, then left knee and feet"
                )
            self.last = int(np.flatnonzero(a["phase"] == "FINISHED")[0])


def preload_pose(model, state, prior, *, loaded=False):
    """Retarget v4's pelvis drop while holding the official robot's feet and elbow contact lines."""
    from elf3_forearm import get_pose, set_pose
    from scipy.optimize import least_squares

    data = mj.MjData(model)
    mj.mj_setState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
    mj.mj_forward(model, data)
    q0 = get_pose(model, data)
    feet = [model.body(s + "_ankle_x_link").id for s in ("l", "r")]
    arms = [model.body(s + "_elbow_y_link").id for s in ("l", "r")]
    pelvis = model.body("waist_z_link").id
    ft = data.xpos[feet].copy()
    fr = Rotation.from_matrix(data.xmat[feet].reshape(2, 3, 3))
    offsets = np.array([[0.08, 0, 0], [0.20, 0, 0]])

    def arm_line():
        return data.xpos[arms, None, :] + np.einsum(
            "sij,pj->spi", data.xmat[arms].reshape(2, 3, 3), offsets
        )

    arm_target = arm_line()
    pelvis_goal = data.xpos[pelvis].copy()
    if loaded:
        # IK setpoints slightly below the surface request pressure through motor torques.
        arm_target[:, :, 2] -= [0.0066, 0.012]
        pelvis_goal += prior.preload_delta
    else:
        pelvis_goal[2] -= prior.drop
    act = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    seed = q0.copy()
    seed[act[prior.motor]] += prior.joints[prior.frame[1] - 1] - prior.joints[prior.start]
    low = np.r_[q0[:3] - 0.22, [-2] * 3, model.jnt_range[1:, 0]]
    high = np.r_[q0[:3] + 0.22, [2] * 3, model.jnt_range[1:, 1]]
    wrist = act[model.actuator("l_wrist_y_joint").id]
    if loaded:
        seed[[int(model.joint(side + "_knee_y_joint").dofadr[0]) for side in ("l","r")]] += .85

    def residual(x):
        set_pose(model, data, x)
        return np.r_[
            150 * (data.xpos[feet] - ft).ravel(),
            12
            * (fr.inv() * Rotation.from_matrix(data.xmat[feet].reshape(2, 3, 3)))
            .as_rotvec()
            .ravel(),
            (300 if loaded else 160) * (arm_line() - arm_target).ravel(),
            80 * (data.xpos[pelvis] - pelvis_goal),
            4 * (x[wrist] - q0[wrist]),
            0.35 * (x - seed),
        ]

    fit = least_squares(
        residual, np.clip(seed, low + 1e-6, high - 1e-6), bounds=(low, high), max_nfev=90
    )
    return fit.x if loaded else fit.x[act]


class PlatformAscent(gym.Env):
    dt = 0.02

    def __init__(self, approach, signature, paths, motion):
        super().__init__()
        with np.load(paths[0], allow_pickle=False) as sample:
            height = float(sample["platform_height_m"]) if "platform_height_m" in sample else 1.0
        self.model = m = approach.build_model("official", True, approach.specification(), height)
        self.signature = signature(m)
        self.handoffs = []
        for path in paths:
            with np.load(path, allow_pickle=False) as a:
                if (
                    str(a["model_signature"]) != self.signature
                    or not json.loads(str(a["report"]))["screening_pass"]
                ):
                    raise ValueError(f"Incompatible or unsuccessful handoff: {path}")
                self.handoffs.append((float(a["yaw"]), a["supported"].copy()))
        self.prior = AscentPrior(motion, m)
        self.preload_poses = [preload_pose(m, state, self.prior) for _, state in self.handoffs]
        self.data = mj.MjData(m)
        j = m.actuator_trnid[:, 0]
        self.qa, self.va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        self.limits = m.jnt_range[j]
        self.ctrl_min, self.ctrl_max = m.actuator_ctrlrange.T
        self.wrist = m.actuator("l_wrist_y_joint").id
        self.active = np.array([i for i in self.prior.motor if i != self.wrist])
        self.kp = np.where(self.ctrl_max >= 90, 165.0, np.where(self.ctrl_max >= 45, 85.0, 35.0))
        self.kp[self.wrist] = 80.0
        self.kd = np.where(self.ctrl_max >= 90, 6.0, np.where(self.ctrl_max >= 45, 3.0, 1.6))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(len(self.active),), dtype=np.float32)
        self.floor = m.geom("ut_infinite_ground").id
        self.table = m.geom("approach_platform").id
        self.grip = m.geom("left_gripper_envelope").id
        self.arm = [m.geom(side + "_elbow_y_collision").id for side in ("l", "r")]
        # The transverse capsule at the knee pivot is attached to the distal thigh.
        # The geom named knee_y_collision lies 19 cm down the shin.
        self.knee = [m.geom(side + "_hip_z_2_collision").id for side in ("r", "l")]
        self.thigh = [m.geom(side + "_hip_z_1_collision").id for side in ("r", "l")]
        shins = [m.geom(side + "_knee_y_collision").id for side in ("r", "l")]
        feet = [m.body(side + "_ankle_x_link").id for side in ("r", "l")]
        self.slot = np.full(m.ngeom, -1, dtype=int)
        self.slot[self.arm + self.knee] = np.arange(4)
        for i, b in enumerate(feet):
            self.slot[m.geom_bodyid == b] = 4 + i
        guards = [m.geom(s + "_shoulder_z_2_collision").id for s in ("l", "r")]
        self.wrist_geom = np.array(
            ["wrist" in m.body(m.geom_bodyid[i]).name and i != self.grip for i in range(m.ngeom)]
        )
        self.slot[shins] = 6  # legal top surface contact, excluded from measured knee loads
        self.slot[guards] = 6  # permissible elbow guards, excluded from forearm force totals
        self.arm_bodies = m.geom_bodyid[self.arm]
        self.arm_motor = np.array(
            [
                i
                for i in self.active
                if any(part in m.actuator(int(i)).name for part in ("shoulder", "elbow", "wrist"))
            ]
        )
        self.surface_offsets = np.array([[0.08, 0, 0], [0.20, 0, 0]])
        self.pelvis = m.body("waist_z_link").id
        self.torso = m.body("torso_link").id
        self.feet = feet
        self.front = approach.PLATFORM_FRONT
        self.top = m.geom_pos[self.table, 2] + m.geom_size[self.table, 2]
        self.substeps = round(self.dt / m.opt.timestep)
        self.action = np.zeros(self.action_space.shape)
        sample, _ = self.reset(seed=0)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=sample.shape, dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get("handoff")
        if index is None:
            index = self.np_random.integers(len(self.handoffs))
        self.yaw, state = self.handoffs[index]
        self.preload_pose = self.preload_poses[index]
        m, d = self.model, self.data
        mj.mj_resetData(m, d)
        mj.mj_setState(m, d, state, mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_forward(m, d)
        self.home = d.qpos[self.qa].copy()
        self.bias = d.ctrl.copy()
        self.command = self.home.copy()
        self.action[:] = 0
        self.stage = self.tick = self.stage_tick = self.hold = self.air = 0
        self.unsupported = 0
        self.peak_knee_load = 0.0
        self.last_ground = np.zeros(2, dtype=int)
        self.reason = ""
        self.target = np.column_stack(
            (
                self.front + self.prior.xy[:, 0],
                d.qpos[1] + self.prior.xy[:, 1],
                np.full(len(SITES), self.top),
            )
        )
        self.foot0 = d.xpos[self.feet, 2].copy()
        self.preload_drop = 0.0
        self.start_height = d.xpos[self.pelvis, 2]
        self.push_tick = -100
        self.best_preload = 0.0
        self.arm_anchors = self.arm_points()
        self.target[:2, :2] = self.arm_anchors.mean(axis=1)[:, :2]
        self.min_knee_gap = self.knee_gap()
        self.max_foot_lift = 0.0
        self.max_unload = 0.0
        return self.observe(), {}

    def joint_prior(self, frame):
        prior = self.prior
        pose = self.home.copy()
        pose[prior.motor] += prior.joints[frame] - prior.joints[prior.start]
        # Apply the official-contact IK correction at the bottom of the squat,
        # fading it out again by the original release frame.
        peak = prior.frame[1] - 1
        weight = np.interp(frame, [prior.start, peak, prior.frame[2]], [0.0, 1.0, 0.0], right=0.0)
        peak_pose = self.home[prior.motor] + prior.joints[peak] - prior.joints[prior.start]
        pose[prior.motor] += weight * (self.preload_pose[prior.motor] - peak_pose)
        pose[self.wrist] = self.home[self.wrist]
        return np.clip(pose, *self.limits.T)

    def observe(self):
        d = self.data
        rot = d.xmat[self.torso].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        horizontal = Rotation.from_euler("z", yaw).as_matrix()
        mask = MASKS[self.stage]
        points = ((self.target - d.qpos[:3]) @ horizontal) * mask[:, None]
        return np.asarray(
            np.r_[
                d.qpos[self.qa],
                d.qvel[self.va] * 0.08,
                rot[2],
                d.qvel[3:6] * 0.1,
                points.ravel(),
                mask,
                np.eye(len(PHASES))[self.stage],
                self.action,
            ],
            dtype=np.float32,
        )

    def knee_gap(self):
        d = self.data
        return float(np.linalg.norm(d.geom_xpos[self.knee[0]] - self.target[2]))

    def contacts(self):
        d = self.data
        pairs = d.contact.geom
        a, b = pairs.T
        # MuJoCo can retain contacts at a positive margin.
        active = d.contact.dist <= 0
        gripper_hit = bool(np.any(active & ((a == self.grip) | (b == self.grip))))
        it = np.flatnonzero(
            active & ((a == self.table) | (b == self.table) | (a == self.floor) | (b == self.floor))
        )
        surface_first = (a[it] == self.table) | (a[it] == self.floor)
        surface = np.where(surface_first, a[it], b[it])
        other = np.where(surface_first, b[it], a[it])
        slot = self.slot[other]
        # The thigh capsule ends at the knee pivot and overlaps its guard. Only
        # the shared patch is allowable and it earns no knee-force credit.
        for thigh, knee in zip(self.thigh, self.knee):
            match = other == thigh
            if np.any(match):
                radius = self.model.geom_size[knee, :2].sum() + .002
                inside = np.linalg.norm(d.contact.pos[it[match]] - d.geom_xpos[knee], axis=1) <= radius
                slot[np.flatnonzero(match)[inside]] = 6
        grounded = surface == self.floor
        normals = d.contact.frame[it, 2] * np.where(surface_first, 1, -1)
        on_top = (normals > 0.8) & (abs(d.contact.pos[it, 2] - self.top) < 0.012)
        bad = bool(np.any(np.where(grounded, (slot < 4) | (slot > 5), (slot < 0) | ~on_top)))
        wrench = np.empty(6)
        loads = np.zeros(8)  # table sites 0..5, floor right/left foot 6..7
        for index, site, is_ground, valid_top in zip(it, slot, grounded, on_top):
            if (is_ground and 4 <= site <= 5) or (not is_ground and valid_top and 0 <= site <= 5):
                mj.mj_contactForce(self.model, d, int(index), wrench)
                loads[site + 2 if is_ground else site] += max(0.0, wrench[0])
        return loads, gripper_hit, bad

    def arm_points(self):
        d = self.data
        return d.xpos[self.arm_bodies, None, :] + np.einsum(
            "sij,pj->spi", d.xmat[self.arm_bodies].reshape(2, 3, 3), self.surface_offsets
        )

    def support_torque(self):
        """Actuated arm impedance during knee transfer. No external force or pose constraint."""
        d, m = self.data, self.model
        result = np.zeros(m.nu)
        jac = np.empty((3, m.nv))
        for i, points in enumerate(self.arm_points()):
            for j, point in enumerate(points):
                mj.mj_jac(m, d, jac, None, point, int(self.arm_bodies[i]))
                error = self.arm_anchors[i, j] - point
                force = np.array([850.0, 850.0, 1800.0]) * error - np.array([28.0, 28.0, 40.0]) * (
                    jac @ d.qvel
                )
                force[2] -= 11.0
                result += jac[:, self.va].T @ np.clip(force, -110.0, 110.0)
        return result

    def step(self, action):
        m, d = self.model, self.data
        prev = self.action.copy()
        self.action[:] = np.clip(action, -1, 1)
        command = self.home.copy()
        command[self.active] += self.action * np.where(
            self.action >= 0,
            self.limits[self.active, 1] - self.home[self.active],
            self.home[self.active] - self.limits[self.active, 0],
        )
        command = np.clip(command, *self.limits.T)
        if self.stage <= 3:
            # Leg and pelvis motion are learned; the teacher initially keeps the elbow contacts.
            command[self.arm_motor] = self.home[self.arm_motor] + 0.18 * (
                command[self.arm_motor] - self.home[self.arm_motor]
            )
        self.command += 0.35 * (command - self.command)
        grip_hit = bad = False
        for tick in range(self.substeps):
            if tick % 5 == 0:
                arm_torque = self.support_torque() if self.stage <= 3 else 0.0
            d.ctrl[:] = np.clip(
                self.bias
                + arm_torque
                + self.kp * (self.command - d.qpos[self.qa])
                - self.kd * d.qvel[self.va],
                self.ctrl_min,
                self.ctrl_max,
            )
            mj.mj_step(m, d)
            # Check transient gripper contacts even if contact ends before the next policy tick.
            grip_hit |= bool(
                np.any((d.contact.dist <= 0) & np.any(d.contact.geom == self.grip, axis=1))
            )
            pairs = d.contact.geom[d.contact.dist <= 0]
            if len(pairs):
                a, b = pairs.T
                surface = (
                    (a == self.table) | (a == self.floor) | (b == self.table) | (b == self.floor)
                )
                if np.any(surface):
                    bad |= bool(np.any(self.wrist_geom[pairs[surface]]))
        loads, hit, unexpected = self.contacts()
        bad |= unexpected
        grip_hit |= hit
        self.tick += 1
        self.stage_tick += 1
        feet = loads[6:] > 8.0
        self.last_ground[feet] = self.tick
        lift = float((d.xpos[self.feet, 2] - self.foot0).min())
        arms = bool(np.all(loads[:2] > 3.0))
        # A supported simultaneous departure from the floor, not a fall or a one-leg step.
        if arms and np.all(feet) and not (grip_hit or bad):
            self.preload_drop = max(self.preload_drop, self.start_height - d.xpos[self.pelvis, 2])
            if (
                self.stage == 1
                and self.preload_drop - (self.start_height - d.xpos[self.pelvis, 2]) > 0.008
                and d.qvel[2] > 0.03
            ):
                self.push_tick = self.tick
        airborne = (
            self.tick - self.push_tick <= 8
            and not np.any(feet)
            and lift > 0.005
            and arms
            and np.ptp(self.last_ground) <= 4
        )
        self.air = self.air + 1 if airborne else 0
        self.unsupported = self.unsupported + 1 if self.stage == 2 and not arms else 0
        if self.stage == 2:
            self.peak_knee_load = max(self.peak_knee_load, loads[2])
        up = float(d.xmat[self.torso, 8])
        stage = self.stage
        stand = (
            np.all(loads[4:6] > 20)
            and d.qpos[2] > self.top + 0.75
            and up > 0.9
            and np.linalg.norm(d.qvel[:3]) < 0.25
        )
        drop = self.start_height - d.xpos[self.pelvis, 2]
        preload = arms and np.all(feet) and drop >= 0.65 * self.prior.drop
        gates = [
            preload,
            self.air >= 2,
            loads[2] > 8 and arms and not np.any(feet),
            np.all(loads[2:4] > 8),
            loads[4] > 15,
            np.all(loads[4:6] > 15),
            stand,
        ]
        self.hold = self.hold + 1 if gates[stage] else 0
        attained = self.hold >= (25 if stage == 6 else 1 if stage == 1 else 3)
        gap = self.knee_gap()
        self.min_knee_gap = min(self.min_knee_gap, gap)
        self.max_foot_lift = max(self.max_foot_lift, lift)
        self.max_unload = (
            max(self.max_unload, min(1.0, 1.0 - loads[6:].sum() / (m.body_mass.sum() * 9.81)))
            if arms
            else self.max_unload
        )
        # Stage one uses joint motion as a soft training prior. The actor receives contact targets.
        prior = self.prior
        end = prior.frame[stage + 1] if stage + 1 < len(PHASES) else prior.last
        time = min(prior.time[end], prior.time[prior.frame[stage]] + self.stage_tick * self.dt)
        frame = min(end, np.searchsorted(prior.time, time))
        goal = self.joint_prior(frame)[prior.motor]
        imitation = np.exp(-5 * np.mean((d.qpos[self.qa[prior.motor]] - goal) ** 2))
        current_pos = np.vstack((d.geom_xpos[self.arm + self.knee], d.xpos[self.feet]))
        mask = MASKS[stage]
        tracking = np.mean(
            np.exp(-6 * np.linalg.norm(current_pos[mask] - self.target[mask], axis=1))
        )
        reward = self.dt * (
            0.8 * (imitation - 0.80) + tracking - 0.60 - 0.12 * np.mean((self.action - prev) ** 2)
        )
        if stage == 0:
            progress = np.clip(drop / prior.drop, 0, 1) if arms and np.all(feet) else 0.0
            reward += 4.0 * (progress - self.best_preload)
            self.best_preload = progress
            reward += self.dt * (1.5 * progress - 0.45 - 0.8 * float(not arms))
        elif stage <= 2:
            reward += self.dt * (float(arms) * (1.0 + 2.0 * np.clip(lift / 0.12, 0, 1)) - 1.1)
        success = False
        if grip_hit or bad or up < -0.35 or d.qpos[2] < 0.5 or not np.isfinite(d.qpos).all():
            self.reason = "gripper_contact" if grip_hit else "unplanned_contact" if bad else "fall"
        elif self.unsupported >= 4:
            self.reason = "lost_forearm_support"
        elif stage <= 2 and loads[3] > 8:
            self.reason = "left_knee_first"
        elif attained:
            if stage == 6:
                success = True
                self.reason = "standing_on_platform"
            else:
                self.stage += 1
                self.stage_tick = self.hold = 0
            reward += 2.0
        if self.reason and not success:
            reward -= 2.0
        truncated = (
            self.tick >= 550 or (stage == 0 and self.stage_tick >= 125)
        ) and not self.reason
        if truncated:
            self.reason = "timeout"
        info = {
            "stage": PHASES[self.stage],
            "stage_index": self.stage,
            "success": success,
            "reason": self.reason,
            "max_bilateral_foot_lift_m": self.max_foot_lift,
            "min_right_knee_target_distance_m": self.min_knee_gap,
            "max_supported_preload_drop_m": float(self.preload_drop),
            "right_knee_load_after_takeoff_N": float(self.peak_knee_load),
            "max_supported_unload_fraction": float(self.max_unload),
        }
        return self.observe(), float(reward), bool(self.reason and not truncated), truncated, info


def imitation_samples(env):
    """Kinematic v4 samples are used only for imitation initialization, never as physics resets."""
    prior, d, m = env.prior, env.data, env.model
    examples, actions = [], []
    for i in range(len(env.handoffs)):
        env.reset(options={"handoff": i})
        root = d.qpos[:3].copy()
        orientation = Rotation.from_quat(d.qpos[3:7], scalar_first=True)
        joint_target = env.joint_prior

        def normalized(pose):
            delta = pose[env.active] - env.home[env.active]
            scale = np.where(
                delta >= 0,
                env.limits[env.active, 1] - env.home[env.active],
                env.home[env.active] - env.limits[env.active, 0],
            )
            return np.clip(delta / np.maximum(1e-6, scale), -1, 1)

        for t in range(prior.start, prior.last):
            d.qpos[:3] = root + prior.root_delta[t]
            d.qpos[3:7] = (orientation * prior.rotation_delta[t]).as_quat(scalar_first=True)
            d.qpos[env.qa] = joint_target(t)
            d.qvel[env.va] = (joint_target(t + 1) - joint_target(t)) / (
                prior.time[t + 1] - prior.time[t]
            )
            mj.mj_forward(m, d)
            env.stage = int(np.searchsorted(prior.frame[1:], t, side="right"))
            env.action[:] = normalized(joint_target(t))
            examples.append(env.observe())
            actions.append(normalized(joint_target(min(t + 3, prior.last))))
    env.reset()
    return np.array(examples), np.array(actions, dtype=np.float32)


def fit_preload(env, seed):
    """Adapt v4's crouch to the loaded official model using only the training handoffs."""
    rng = np.random.default_rng(seed)
    groups = ("waist_y", "hip_y", "knee_y", "ankle_y", "shoulder_y", "elbow_y")
    basis = np.array([[g in env.model.actuator(int(i)).name for g in groups] for i in env.active])
    centre = min(range(len(env.handoffs)), key=lambda i: abs(env.handoffs[i][0]))

    def rollout(offsets, index, collect=False):
        obs, _ = env.reset(options={"handoff": index})
        delta = (env.preload_pose - env.home)[env.active] + basis @ offsets
        span = np.where(
            delta >= 0,
            env.limits[env.active, 1] - env.home[env.active],
            env.home[env.active] - env.limits[env.active, 0],
        )
        target = np.clip(delta / np.maximum(1e-6, span), -1.0, 1.0)
        samples, actions = [], []
        value = 0.0
        for step in range(45):
            u = min(1.0, (step + 1) / 24)
            command = target * u * u * (3.0 - 2 * u)
            before = obs
            obs, _, done, timeout, info = env.step(command)
            if done or timeout:
                break
            if collect:
                samples.append(before)
                actions.append(command.copy())
            if env.stage > 0:
                value += 2.0
                break
            loads = env.contacts()[0]
            if np.all(loads[:2] > 3.0) and np.all(loads[6:] > 8.0):
                value += (
                    np.clip(
                        (env.start_height - env.data.xpos[env.pelvis, 2]) / env.prior.drop, 0.0, 1.0
                    )
                    / 45
                )
        return value, samples, actions, info

    mean = np.zeros(len(groups))
    std = np.array([0.20, 0.35, 0.50, 0.23, 0.17, 0.20])
    best, best_score = mean.copy(), -np.inf
    for _ in range(12):
        proposals = rng.normal(mean, std, (32, len(groups)))
        scores = np.array([rollout(params, centre)[0] for params in proposals])
        order = np.argsort(scores)
        if scores[order[-1]] > best_score:
            best, best_score = proposals[order[-1]].copy(), float(scores[order[-1]])
        elite = proposals[order[-7:]]
        mean, std = elite.mean(axis=0), 0.83 * elite.std(axis=0) + 0.015
    x, y, reports = [], [], []
    for i in range(len(env.handoffs)):
        _, samples, actions, info = rollout(best, i, collect=True)
        x.extend(samples)
        y.extend(actions)
        reports.append(dict(info, yaw_deg=env.yaw))
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        dict(
            joint_groups=groups,
            offset_rad=best.tolist(),
            yaw_deg=env.handoffs[centre][0],
            rollouts=reports,
        ),
    )


def evaluate(agent, env):
    results = []
    for i in range(len(env.handoffs)):
        ob, _ = env.reset(options={"handoff": i})
        for k in range(550):
            action, _ = agent.predict(ob, deterministic=True)
            ob, _, end, timeout, info = env.step(action)
            if end or timeout:
                results.append(dict(info, yaw_deg=env.yaw, duration_s=(k + 1) * env.dt))
                break
    return results


def experiment(args, approach, signature):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(1)
    env = PlatformAscent(approach, signature, args.handoffs, args.motion)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    contract = {"model": env.signature, "reference": env.prior.sha, "version": 4}
    if args.cmd == "evaluate" or args.resume:
        checkpoint = args.checkpoint if args.cmd == "evaluate" else args.resume
        meta = json.loads((checkpoint.parent / "contract.json").read_text())
        if contract != meta:
            raise ValueError("Checkpoint does not match robot or motion prior")
        agent = PPO.load(checkpoint, env=env, device=args.device)
    else:
        agent = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=0.00005,
            target_kl=0.012,
            seed=args.seed,
            verbose=0,
            device=args.device,
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "log_std_init": -3.7},
        )
    print(f"PPO device: {agent.device} | MuJoCo physics: CPU", flush=True)
    if args.cmd == "evaluate":
        result = evaluate(agent, env)
        (out / "evaluation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    (out / "contract.json").write_text(json.dumps(contract, indent=2))
    meta = {
        "reference_file": str(env.prior.path),
        "device": str(agent.device),
        "handoffs": [str(p) for p in args.handoffs],
        "yaw_deg": [y for y, _ in env.handoffs],
        "seed": args.seed,
        "requested_steps": args.steps,
        "initialization": "resume" if args.resume else "v4 pose imitation",
        "stage": "contact-conditioned ascent tracker, oracle platform and ordered contacts",
        "wrist": "left wrist held at measured flipped handoff; wrist and gripper are not support sites",
        "platform_height_m": float(env.top),
        "extra_gripper_mass_kg": 0.0,
    }
    if not args.resume:
        obs, actions = imitation_samples(env)
        physical_obs, physical_actions, meta["preload_fit"] = fit_preload(env, args.seed)
        if len(physical_obs):
            obs = np.concatenate([obs, np.tile(physical_obs, (4, 1))])
            actions = np.concatenate([actions, np.tile(physical_actions, (4, 1))])
        x = torch.as_tensor(obs, device=agent.device)
        y = torch.as_tensor(actions, device=agent.device)
        params = list(agent.policy.mlp_extractor.policy_net.parameters()) + list(
            agent.policy.action_net.parameters()
        )
        optim = torch.optim.Adam(params, lr=0.001)
        for _ in range(100):
            for idx in torch.randperm(len(x), device=agent.device).split(128):
                predicted = agent.policy.get_distribution(x[idx]).distribution.mean
                loss = (predicted - y[idx]).square().mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
        with torch.no_grad():
            meta["imitation_mse"] = float(
                (agent.policy.get_distribution(x).distribution.mean - y).square().mean()
            )
        meta["imitation_samples"] = len(x)
        agent.save(out / "imitation")
    meta["before_ppo"] = evaluate(agent, env)
    print("Training initialized", json.dumps(meta["before_ppo"]), flush=True)

    evaluation_env = PlatformAscent(approach, signature, args.handoffs, args.motion)

    def score(rows):
        return (
            sum(r["success"] for r in rows),
            sum(r["stage_index"] for r in rows),
            sum(r["reason"] == "timeout" for r in rows),
            min(r["max_supported_preload_drop_m"] for r in rows),
        )

    best_score = score(meta["before_ppo"])
    agent.save(out / "best")
    meta["best_step"] = int(agent.num_timesteps)
    meta["best_training_eval"] = meta["before_ppo"]

    class Record(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episodes = []

        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    self.episodes.append(
                        {k: v for k, v in info.items() if k in meta["before_ppo"][0]}
                    )
            if self.n_calls % 2048 == 0:
                self.select_best()
            return True

        def select_best(self):
            nonlocal best_score
            rows = evaluate(self.model, evaluation_env)
            candidate = score(rows)
            if candidate > best_score:
                best_score = candidate
                self.model.save(out / "best")
                meta["best_step"] = int(self.model.num_timesteps)
                meta["best_training_eval"] = rows

    record = Record()
    agent.learn(
        total_timesteps=args.steps, callback=record, reset_num_timesteps=not bool(args.resume)
    )
    agent.save(out / "policy")
    record.select_best()
    meta.update(
        timesteps=int(agent.num_timesteps),
        after_ppo=evaluate(agent, env),
        completed_training_episodes=len(record.episodes),
    )
    (out / "training.json").write_text(json.dumps(meta, indent=2))
    (out / "episodes.json").write_text(json.dumps(record.episodes))
    print(json.dumps(meta["after_ppo"], indent=2), flush=True)
