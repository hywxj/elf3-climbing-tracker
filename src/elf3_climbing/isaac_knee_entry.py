"""PhysX geometry and measured-contact sequence for the elbow-supported knee entry."""

import numpy as np
import torch


WRIST_SUPPORT_N = 3.0
WRIST_SUPPORT_LIMIT_S = 0.10
PLANTED_SLIP_M_S = 0.12
ASSIST_WINDOW_S = 0.20


# On the v4 0.70 m reference mesh the loaded knee surface is 0.096 m from its axis.
KNEE_PATCH_RADIUS_M = 0.11


def top_face_points(contact_data, box, top_heights):
    """Read normal forces at individual PhysX contact points on a horizontal box."""
    forces, points, normals, _, counts, starts = contact_data
    counts = counts[:, box].long()
    offsets = counts.cumsum(0) - counts
    rows = torch.arange(counts.numel(), device=counts.device).repeat_interleave(counts)
    indices = (
        starts[rows, box].long() + torch.arange(rows.numel(), device=rows.device) - offsets[rows]
    )
    xyz, nz = points[indices], normals[indices, 2]
    keep = (nz > 0.8) & ((xyz[:, 2] - top_heights[rows]).abs() < 0.012)
    load = forces.reshape(-1)[indices].clamp_min(0)
    return rows, xyz, nz, keep, load


def tabletop_loads(contact_data, box, top_heights, knee_pivots=None):
    """Sum individual PhysX normal forces on the top face and flag other loaded patches."""
    rows, xyz, nz, keep, load = top_face_points(contact_data, box, top_heights)
    if knee_pivots is not None:
        keep &= (xyz - knee_pivots[rows]).norm(dim=1) <= KNEE_PATCH_RADIUS_M
    support = torch.zeros_like(top_heights).index_add_(0, rows, torch.where(keep, load * nz, 0.0))
    rejected = torch.zeros_like(top_heights).index_add_(0, rows, torch.where(keep, 0.0, load))
    return support, rejected


def foot_patch_motion(contact_data, box, heights, center, linear, angular, outward):
    """Force-weighted tangential speed and signed outward speed on the stationary ground."""
    rows, xyz, nz, keep, normal = top_face_points(contact_data, box, heights)
    weight = torch.where(keep, normal * nz, 0.0)
    velocity = linear[rows] + torch.cross(angular[rows], xyz - center[rows], dim=1)
    xy = velocity[:, :2]

    def per_foot(value):
        return torch.zeros_like(heights).index_add_(0, rows, value)

    load = per_foot(weight)
    slip = per_foot(weight * xy.norm(dim=1)) / load.clamp_min(1e-6)
    lateral = per_foot(weight * (xy * outward).sum(1)) / load.clamp_min(1e-6)
    return load, slip, lateral


def add_knee_colliders(stage, robot_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    for side in ("l", "r"):
        name = f"{side}_knee_y_link"
        body = stage.GetPrimAtPath(f"{robot_path}/{name}")
        children = list(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        if any(p.HasAPI(UsdPhysics.CollisionAPI) for p in children):
            continue
        (visual,) = [p for p in children if p.IsA(UsdGeom.Mesh)]
        src = UsdGeom.Mesh(visual)
        mesh = UsdGeom.Mesh.Define(stage, body.GetPath().AppendChild("ClimbingKneeHull"))
        mesh.CreatePointsAttr(src.GetPointsAttr().Get())
        mesh.CreateFaceVertexCountsAttr(src.GetFaceVertexCountsAttr().Get())
        mesh.CreateFaceVertexIndicesAttr(src.GetFaceVertexIndicesAttr().Get())
        mesh.AddTransformOp().Set(cache.ComputeRelativeTransform(visual, body)[0])
        mesh.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        # Same-leg mechanical neighbours overlap the knee mesh at the joint.
        pairs = UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel()
        for adjacent in ("hip_z_link", "ankle_x_link"):
            pairs.AddTarget(f"{robot_path}/{side}_{adjacent}")


class KneeContacts:
    def __init__(self, env):
        self.env = env
        ref = env.reference_np
        tops = ref["terrain_box_centers_w"][:, 2] + ref["terrain_box_half_sizes"][:, 2]
        self.upper, self.lower = int(tops.argmax()), int(tops.argmin())
        self.floor_z = float(tops[self.lower])
        self.floor_heights = env.scene.env_origins[:, 2] + self.floor_z
        self.top_heights = env.scene.env_origins[:, 2] + float(tops[self.upper])
        self.takeoff = env.segment_start + int(
            np.flatnonzero(
                ~ref["contact_mask"][env.segment_start : env.segment_end + 1, :2].any(1)
            )[0]
        )
        # Start of extension after the lowest torso pose in the grounded reference.
        grounded_z = ref["root_pos"][env.segment_start : self.takeoff + 1, 2]
        self.push_start = env.segment_start + int(np.argmin(grounded_z))
        self.support_step = torch.full_like(env.starts, -100)
        self.push_step = self.support_step.clone()
        self.planted_step = self.support_step.clone()
        self.planted_streak = torch.zeros_like(env.starts)
        self.assist_window = max(1, round(ASSIST_WINDOW_S / env.step_dt))
        masses = env.robot.data.default_mass.to(device=env.starts.device)
        self.total_mass = masses.sum(1)
        self.mass_fraction = masses / self.total_mass[:, None]
        self.assisted = env.cfg.knee_support_rule == "assisted-push"
        self.trace_edge = None
        self.hip_ids = env.robot.find_bodies(["l_hip_y_link", "r_hip_y_link"], preserve_order=True)[0]
        if self.assisted:
            self.support_streak = torch.zeros_like(env.starts)
            self.unsupported_steps = torch.zeros_like(env.starts)
            self.knee_load_steps = torch.zeros_like(env.starts)
            self.reground_steps = torch.zeros_like(env.starts)
            self.knee_landing_observed = torch.zeros_like(env.starts, dtype=torch.bool)
            self.transfer_lost = torch.zeros_like(env.starts, dtype=torch.bool)
            self.pushed_this_episode = torch.zeros_like(self.transfer_lost)
            self.lift_step = torch.full_like(env.starts, -100)
        self.transfer_guide = self.assisted and getattr(env.cfg, "knee_transfer_guide", None) is not None
        self.wrist_steps = torch.zeros_like(env.starts)
        self.wrist_limit = max(1, round(WRIST_SUPPORT_LIMIT_S / env.step_dt))
        self.flight = torch.zeros_like(env.starts, dtype=torch.bool)
        self.pending_lift = self.flight.clone()
        self.assumed = self.flight.clone()
        self.observed = self.flight.clone()
        self.arm_names = ref["site_bodies"][2:4].tolist()
        self.foot_names = ref["site_bodies"][:2].tolist()
        self.knee_names = ref["site_bodies"][6:8].tolist()

    def enable_pose_trace(self):
        from scipy.spatial.transform import Rotation

        env = self.env
        ref = env.reference_np
        center = ref["terrain_box_centers_w"][self.upper]
        half = ref["terrain_box_half_sizes"][self.upper]
        axes = Rotation.from_quat(ref["terrain_box_quat_wxyz"][self.upper], scalar_first=True).as_matrix()
        start = (ref["root_pos"][env.segment_start] - center) @ axes
        axis = int(np.argmax(abs(start[:2]) / half[:2]))
        inward = -np.sign(start[axis]) * axes[:, axis]
        self.trace_inward = torch.as_tensor(inward, dtype=self.top_heights.dtype, device=env.starts.device)
        edge = center - half[axis] * inward
        self.trace_edge = torch.as_tensor(edge, dtype=self.top_heights.dtype, device=env.starts.device)

    def reset(self, ids):
        self.support_step[ids] = -100
        self.push_step[ids] = -100
        self.planted_step[ids] = -100
        self.planted_streak[ids] = 0
        if self.assisted:
            self.support_streak[ids] = 0
            self.unsupported_steps[ids] = 0
            self.knee_load_steps[ids] = 0
            self.reground_steps[ids] = 0
            self.knee_landing_observed[ids] = False
            self.transfer_lost[ids] = False
            self.pushed_this_episode[ids] = False
            self.lift_step[ids] = -100
        self.wrist_steps[ids] = 0
        self.assumed[ids] = self.env.starts[ids] >= self.takeoff
        self.flight[ids] = self.assumed[ids]
        self.observed[ids] = False
        self.pending_lift[ids] = False

    def loads(self, names, box):
        return torch.stack(
            [self.env.support_sensors[n].data.force_matrix_w[:, 0, box, 2] for n in names], dim=1
        )

    def surface_loads(self, body, knee_pivots=None):
        sensor = self.env.support_sensors[body]
        detail = sensor.contact_physx_view.get_contact_data(dt=self.env.cfg.sim.dt)
        return tabletop_loads(detail, self.upper, self.top_heights, knee_pivots)

    def ground_motion(self):
        e = self.env
        d = e.robot.data
        forward = e.advance_direction
        left = torch.stack((-forward[1], forward[0]))
        patches = []
        for body, name, direction in zip(e.site_ids[:2], self.foot_names, (left, -left)):
            detail = e.support_sensors[name].contact_physx_view.get_contact_data(dt=e.cfg.sim.dt)
            patches.append(foot_patch_motion(
                detail, self.lower, self.floor_heights, d.body_com_pos_w[:, body],
                d.body_com_lin_vel_w[:, body], d.body_com_ang_vel_w[:, body], direction))
        return tuple(torch.stack(values, dim=1) for values in zip(*patches))

    def measure(self, i, feet, *, require_reference_goal=True, recover_supported_lift=False):
        e = self.env
        # A toe brush can break the instantaneous flight bit. Recovery requires a
        # previous physical launch, another clear rising sample, and bilateral arm support.
        recover_supported_lift = recover_supported_lift or self.transfer_guide
        arms = torch.stack([self.surface_loads(name)[0] for name in self.arm_names], dim=1)
        sites = e.site_positions()
        right_patch, other_right = self.surface_loads(self.knee_names[1], sites[:, 7])
        knees = self.loads(self.knee_names, self.upper)
        foot_load = self.loads(self.foot_names, self.lower)
        def wrist_vectors(axis):
            return torch.stack(
                [e.support_sensors[f"{side}_wrist_{axis}_link"].data.force_matrix_w[:, 0, self.upper]
                 for side in ("l", "r")], dim=1)

        # The wrist_x collision mesh is the long forearm shell. The two
        # compact wrist_y and wrist_z meshes meet at the distal joint.
        y_vectors, z_vectors = wrist_vectors("y"), wrist_vectors("z")
        wrist_y_force = y_vectors.norm(dim=2)
        wrist_forces = z_vectors.norm(dim=2)
        wrists = z_vectors[:, :, 2]
        wrist_mode = e.cfg.knee_wrist_contact
        forearm_transfer = wrist_mode == "tabletop-touch"
        if wrist_mode != "distal-only":
            wrist_forces = wrist_forces + wrist_y_force
            wrists = wrists + y_vectors[:, :, 2]
        arm_support = arms
        wrist_forbidden = (wrist_forces > WRIST_SUPPORT_N).any(1)
        wrist_top = None
        if wrist_mode in ("tabletop", "tabletop-touch"):
            wrist_faces = [self.surface_loads(f"{side}_wrist_{axis}_link")
                           for side in ("l", "r") for axis in ("y", "z")]
            wrist_top = torch.stack([v[0] for v in wrist_faces], dim=1).reshape(-1, 2, 2).sum(2)
            rejected = torch.stack([v[1] for v in wrist_faces], dim=1)
            wrist_forbidden = (rejected > WRIST_SUPPORT_N).any(1)
            if wrist_mode == "tabletop":
                arm_support = arms + wrist_top
        arms_on = (arm_support > 3.0).all(1)
        feet_loaded = (foot_load > 8.0).all(1)
        wrist_load = (wrist_forces > WRIST_SUPPORT_N).any(1)
        self.wrist_steps.copy_(torch.where(wrist_forbidden, self.wrist_steps + 1, 0))
        sustained_wrist = self.wrist_steps >= self.wrist_limit
        patch_load, slip, outward = self.ground_motion()
        com_vz = (e.robot.data.body_com_lin_vel_w[:, :, 2] * self.mass_fraction).sum(1)
        phased = e.cfg.knee_support_rule != "continuous"
        double_support = arms_on & feet_loaded & ~wrist_forbidden
        loaded = patch_load > 8.0
        sliding = (loaded & (slip > PLANTED_SLIP_M_S)).any(1)
        planted = loaded.all(1) & ~sliding & ~wrist_forbidden
        self.planted_step.copy_(torch.where(planted, e.ref_steps, self.planted_step))
        self.planted_streak.copy_(torch.where(planted, self.planted_streak + 1, 0))
        initial_support = arms_on & planted if phased else double_support
        self.support_step.copy_(torch.where(initial_support, e.ref_steps, self.support_step))
        # Load on the weaker foot as a fraction of total robot weight.
        weaker_foot_bw = patch_load.amin(1) / (self.total_mass * 9.81)
        if phased:
            recent_assist = e.ref_steps - self.support_step <= self.assist_window
            pushed = recent_assist & (self.planted_streak >= 2) & (weaker_foot_bw >= 0.08) & (com_vz > 0.08)
            if self.assisted:
                hip_v = e.robot.data.body_link_lin_vel_w[:, self.hip_ids].mean(1)
                hip_forward = (hip_v[:, :2] * e.advance_direction).sum(1)
                self.support_streak.copy_(torch.where(initial_support, self.support_streak + 1, 0))
                pushed = ((self.support_streak >= 2) & (weaker_foot_bw >= 0.08)
                          & (com_vz > 0.08) & (hip_v[:, 2] > 0.05) & (hip_forward > 0.025)
                          & (i >= self.push_start))
            self.push_step.masked_fill_(sliding | wrist_forbidden, -100)
        else:
            pushed = double_support & (e.robot.data.root_link_lin_vel_w[:, 2] > 0.03)
        self.push_step.copy_(torch.where(pushed, e.ref_steps, self.push_step))
        foot_top = self.loads(self.foot_names, self.upper)
        released = (foot_load < 3.0).all(1) & (foot_top < 3.0).all(1)
        # A two-control-step window permits 40 ms between the last double support and release.
        last_stance = self.planted_step if phased else self.support_step
        eligible = ~wrist_forbidden & (knees[:, 0] < 8)
        if not phased:
            eligible &= arms_on & (knees[:, 1] < 8)
        elif recover_supported_lift or forearm_transfer:
            eligible &= arms_on
        push_age_limit = 2 if self.assisted else 8
        lift = (released & eligible & (e.ref_steps - last_stance <= 2)
                & (e.ref_steps - self.push_step <= push_age_limit))
        clear = (feet[:, :, 2] > self.floor_z + 0.01).all(1)
        self.pending_lift.copy_((self.pending_lift | lift) & released & eligible)
        self.pending_lift &= (e.ref_steps - last_stance <= 8)
        verified_lift = self.pending_lift & clear
        if recover_supported_lift:
            # A short brush of the floor during the launch can break the instantaneous
            # flight flag. Reestablish it only after another measured airborne sample.
            self.lift_step.copy_(torch.where(verified_lift & ~self.observed, e.ref_steps, self.lift_step))
            resumed = (self.observed & (e.ref_steps - self.lift_step <= round(0.20 / e.step_dt))
                       & released & clear & arms_on & eligible & ~self.transfer_lost & (com_vz > 0.03))
            verified_lift |= resumed
        self.observed |= verified_lift
        self.flight.copy_((self.flight | verified_lift) & released)
        local_sites = sites - e.scene.env_origins[:, None]
        knee_delta = local_sites[:, 7] - e.ref["site_pos"][e.segment_end, 7]
        goal_distance = knee_delta.norm(dim=1)
        # The original tracking benchmark scores proximity to its demonstrated landmark.
        # The terrain task checks all four physical tabletop edges at the call site.
        at_target = goal_distance < 0.10 if require_reference_goal else torch.ones_like(self.flight)
        wrong_knee = knees[:, 0] > 8.0
        supported = (
            self.flight
            & clear
            & (right_patch > 8.0)
            & (other_right <= 8.0)
            & at_target
            & ~wrong_knee
            & ~wrist_forbidden
        )
        if self.assisted:
            # During the transfer, accept a brief contact-solver gap. A sustained
            # unsupported fall followed by hitting the platform cannot count as a climb.
            self.pushed_this_episode |= pushed
            carry = (arm_support.sum(1) / (self.total_mass * 9.81 * 0.08)).clamp(0, 1)
            if forearm_transfer:
                # A heavily loaded single forearm cannot replace its unloaded partner.
                carry = torch.minimum(carry, (arms.amin(1) / 3.0).clamp(0, 1))
            self.knee_load_steps.copy_(torch.where(supported, self.knee_load_steps + 1, 0))
            knee_bearing = self.knee_load_steps >= 2
            if forearm_transfer:
                self.knee_landing_observed |= knee_bearing
                replanted = self.observed & ~self.knee_landing_observed & loaded.any(1)
                self.reground_steps.copy_(torch.where(replanted, self.reground_steps + 1, 0))
                self.transfer_lost |= self.reground_steps >= 2
            transferring = self.flight
            lacks_support = carry < 0.5
            if recover_supported_lift or forearm_transfer:
                transferring = self.flight | self.observed
                lacks_support |= ~arms_on
            self.unsupported_steps.copy_(torch.where(
                transferring & ~knee_bearing & lacks_support, self.unsupported_steps + 1, 0))
            self.transfer_lost |= self.unsupported_steps * e.step_dt > 0.06
            supported &= knee_bearing & ~self.transfer_lost
        if not phased:
            supported &= arms_on
        self.metrics = {
            "foot_ground_patch_N": patch_load,
            "foot_ground_slip_m_s": slip,
            "foot_ground_outward_m_s": outward,
            "com_vz_m_s": com_vz,
            "weaker_foot_bodyweight": weaker_foot_bw,
            "bilateral_push": pushed,
            "forearm_assist_recent": e.ref_steps - self.support_step <= self.assist_window,
            "forearm_top_load_N": arms,
            "arm_support_top_N": arm_support,
            "wrist_forbidden": wrist_forbidden,
            "forearm_target_error_m": (local_sites[:, 2:4] - e.ref["site_pos"][i, 2:4]).norm(dim=2),
            "foot_ground_load_N": foot_load,
            "max_foot_load_N": torch.maximum(foot_load, foot_top).amax(1),
            "min_foot_clearance_m": feet[:, :, 2].amin(1) - self.floor_z,
            "right_knee_goal_distance_m": goal_distance,
            "right_knee_top_load_N": right_patch,
            "right_knee_off_patch_load_N": other_right,
            "wrist_top_load_N": wrists,
            "wrist_platform_force_N": wrist_forces,
            "wrist_y_platform_force_N": wrist_y_force,
            "wrist_loaded": wrist_load,
            "wrist_support_failure": sustained_wrist,
            "left_knee_early": wrong_knee,
            "right_knee_early": (knees[:, 1] > 8.0) & ~self.flight,
            "right_knee_supported": supported,
            "takeoff_observed": self.observed.clone(),
            "takeoff_assumed": self.assumed.clone(),
        }
        if wrist_top is not None:
            self.metrics["wrist_tabletop_load_N"] = wrist_top
        if self.assisted:
            self.metrics.update(push_started=self.pushed_this_episode.clone(),
                                arm_carry_fraction=carry, knee_bearing=knee_bearing,
                                transfer_lost=self.transfer_lost.clone())
            self.metrics["push_release_window"] = (e.ref_steps - self.push_step <= push_age_limit)
            self.metrics.update(hip_forward_m_s=hip_forward, hip_vz_m_s=hip_v[:, 2],
                                simultaneous_support=initial_support,
                                support_streak=self.support_streak.clone())
        if self.trace_edge is not None:
            hip_z = e.robot.data.body_link_pos_w[:, self.hip_ids, 2].mean(dim=1)
            self.metrics["hip_height_above_platform_m"] = hip_z - self.top_heights
            self.metrics["right_knee_inside_edge_m"] = (local_sites[:, 7] - self.trace_edge) @ self.trace_inward
        return supported, wrong_knee | self.metrics["right_knee_early"] | sustained_wrist
