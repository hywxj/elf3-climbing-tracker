"""Joint targets for the recovery from a physically reached right-knee handoff."""

import torch

from .knee_transfer import point_jacobian


class MountControl:
    """Track leg clearance while using the loaded limbs to advance and raise the body."""

    parameter_names = ("tempo", "forward_shift_m", "position_feedback", "arm_feedback",
                       "foot_clearance_m", "pose_blend_frames")

    def __init__(self, env):
        self.env = env
        phase = env.reference_np["phase"].tolist()
        self.start = phase.index("LEFT_KNEE_CLEAR") - 1
        self.kneeling = phase.index("KNEEL_UP")
        self.right_swing = phase.index("RIGHT_FOOT_CLEAR")
        self.left_swing = phase.index("LEFT_FOOT_CLEAR")
        self.crouch = phase.index("CROUCH")
        self.stand = phase.index("STAND")
        self.last = len(phase) - 1
        self.parameters = torch.tensor([0.75, 0.08, 0.65, 0.75, 0.06, 85.0],
                                       device=env.device).repeat(env.num_envs, 1)
        self.start_targets = torch.zeros((env.num_envs, 29), device=env.device)
        self.start_sites = torch.zeros((env.num_envs, 8, 3), device=env.device)
        self.start_root = torch.zeros((env.num_envs, 3), device=env.device)
        self.forward = torch.nn.functional.pad(env.advance_direction, (0, 1))
        self.regularizer = 0.075**2 * torch.eye(3, device=env.device)
        names = env.reference_np["joint_names"].tolist()
        self.limbs = []
        for side in ("l", "r"):
            cols = [j for j, name in enumerate(names) if name.startswith(side + "_")
                    and any(part in name for part in ("hip_", "knee_", "ankle_"))]
            joints = [env.joint_ids[j] for j in cols]
            self.limbs.append((cols, joints, [j + 6 for j in joints]))
        for side in ("l", "r"):
            cols = [j for j, name in enumerate(names) if name.startswith(side + "_")
                    and any(part in name for part in ("shoulder_", "elbow_"))]
            joints = [env.joint_ids[j] for j in cols]
            self.limbs.append((cols, joints, [j + 6 for j in joints]))

    def desired(self, entry_target, residual, active, age):
        e = self.env
        data = e.robot.data
        origin = e.scene.env_origins
        sites = e.site_positions() - origin[:, None]
        root = data.root_link_pos_w - origin
        capture = ~active | (age == 0)
        self.start_targets.copy_(torch.where(capture[:, None], entry_target, self.start_targets))
        self.start_sites.copy_(torch.where(capture[:, None, None], sites, self.start_sites))
        self.start_root.copy_(torch.where(capture[:, None], root, self.start_root))
        tempo, shift, leg_feedback, arm_feedback, clearance, blend_length = self.parameters.unbind(1)
        elapsed = (age.clamp_min(0).float() * tempo).clamp(max=self.last-self.start)
        frame = (self.start + elapsed.long()).clamp(max=self.last)
        nxt = (frame + 1).clamp(max=self.last)
        interp = elapsed.frac()[:, None]
        q = torch.lerp(e.ref["joint_pos"][frame], e.ref["joint_pos"][nxt], interp)
        reference_sites = torch.lerp(e.ref["site_pos"][frame], e.ref["site_pos"][nxt], interp[:, None])
        reference_root = torch.lerp(e.ref["root_pos"][frame], e.ref["root_pos"][nxt], interp)
        blend = (elapsed / blend_length.clamp_min(1)).clamp(0, 1)
        blend = blend.square() * (3 - 2 * blend)
        target = q + (self.start_targets - e.ref["joint_pos"][self.start]) * (1 - blend[:, None])
        target += 0.18 * residual.clamp(-1, 1) * (elapsed / 12).clamp(0, 1)[:, None]

        goal = reference_sites + (self.start_sites - e.ref["site_pos"][self.start]) * (1-blend[:, None, None])
        root_goal = reference_root + (self.start_root - e.ref["root_pos"][self.start]) * (1-blend[:, None])
        advance = shift[:, None] * blend[:, None] * self.forward
        root_goal += advance
        goal[:, :2] += advance[:, None]
        goal[:, 6:8] += advance[:, None]
        # During each crossing raise the foot first. A forward target alone can drive the
        # hanging shin into the front face even though the ankle's final pose is reachable.
        for side, begin, end in ((0, self.start, self.kneeling),
                                 (1, self.right_swing, self.left_swing),
                                 (0, self.left_swing, self.stand)):
            u = ((self.start + elapsed - begin) / (end - begin)).clamp(0, 1)
            goal[:, side, 2] += clearance * (torch.sin(torch.pi*u)).square()
        root_correction = (0.55 * (root_goal-root) - 0.055 * data.root_link_lin_vel_w).clamp(-0.12, 0.12)
        jac = e.robot.root_physx_view.get_jacobians()
        for limb, (cols, joints, dofs) in enumerate(self.limbs):
            # As the body rises, keep the knee on its support until that leg swings
            # onto its sole. The opposite leg is free to follow its clearance arc.
            if limb < 2:
                knee_phase = (frame < (self.left_swing if limb == 0 else self.right_swing))
                knee_weight = knee_phase * ((frame - self.start) / 60).clamp(0, 1) if limb == 0 else knee_phase
                foot_weight = 1.0 - knee_weight.float()
                body = e.site_ids[limb]
                foot_j = point_jacobian(jac[:, body, :, dofs], sites[:, limb] + origin - data.body_com_pos_w[:, body])
                kbody = e.site_ids[6+limb]
                knee_j = point_jacobian(jac[:, kbody, :, dofs], sites[:, 6+limb]+origin-data.body_com_pos_w[:, kbody])
                # Blend the Cartesian tasks only over the brief transition between sites.
                j = foot_j * foot_weight[:, None, None] + knee_j * (1-foot_weight[:, None, None])
                error = ((goal[:, limb] - sites[:, limb]) * foot_weight[:, None]
                         + (goal[:, 6+limb] - sites[:, 6+limb]) * (1-foot_weight[:, None]))
                planted = ((e.ref["contact_mask"][frame, limb] > 0) |
                           ((e.ref["contact_mask"][frame, 6+limb] > 0) & knee_phase))
                error -= root_correction * planted[:, None]
                weight = leg_feedback
            else:
                body = e.site_ids[limb]
                j = point_jacobian(jac[:, body, :, dofs], sites[:, limb]+origin-data.body_com_pos_w[:, body])
                error = goal[:, limb] - sites[:, limb] - root_correction
                weight = arm_feedback * (1 - ((frame-self.kneeling)/65).clamp(0, 1))
            current = data.joint_pos[:, joints]
            dq = (target[:, cols] - current).unsqueeze(-1)
            correction, _ = torch.linalg.solve_ex(j @ j.mT + self.regularizer,
                                                   error.clamp(-0.20, 0.20).unsqueeze(-1) - j @ dq)
            target[:, cols] += (weight * (elapsed/12).clamp(0,1))[:, None] * (j.mT @ correction).squeeze(-1).clamp(-0.8, 0.8)
        target[:, e.wrist_joint] = e.fixed_wrist_target
        return torch.where(active[:, None], target, entry_target)


class KneeMountControl(MountControl):
    """Plant the hanging knee while the other knee and wrists still carry the robot."""

    transfer_parameter_names = ("swing_duration_s", "knee_inset_m", "knee_clearance_m",
                                "body_advance_m", "pull_gain_N_m", "landing_height_m")

    def __init__(self, env):
        super().__init__(env)
        self.transfer_parameters = torch.tensor([1.0, 0.13, 0.17, 0.14, 450., 0.043],
                                                device=env.device).repeat(env.num_envs, 1)
        self.entry_pose = torch.zeros_like(self.start_targets)
        self.bias = torch.zeros_like(self.start_targets)
        self.bilateral_ticks = torch.zeros_like(env.starts)
        self.recover_at = torch.full_like(env.starts, -1)
        self.recovery = MountControl(env)
        self.recovery.start = env.reference_np["phase"].tolist().index("BOTH_KNEES") + 8
        self.recovery.parameters[:] = self.parameters.new_tensor([.93, .025, 1.05, .99, .075, 86.])
        self.parameters = self.recovery.parameters
        # Once the wrists are loaded they are better anchors than the proximal shell
        # markers. Use the distal wrist position when controlling the shoulder and elbow.
        self.hand_bodies = env.robot.find_bodies(["l_wrist_y_link", "r_wrist_y_link"],
                                                  preserve_order=True)[0]
        self.hand_anchors = self.start_sites[:, :2].clone()

    @staticmethod
    def _smooth(u):
        x = u.clamp(0, 1)
        return x.square() * (3.0 - 2.0*x)

    def desired(self, entry_target, residual, active, age):
        env = self.env
        data = env.robot.data
        origin = env.scene.env_origins
        if env.knee_contacts.trace_edge is None:
            env.knee_contacts.enable_pose_trace()
        direction = env.knee_contacts.trace_inward
        top = env.knee_contacts.top_heights - origin[:, 2]
        edge = env.knee_contacts.trace_edge
        sites = env.site_positions() - origin[:, None]
        hands = data.body_link_pos_w[:, self.hand_bodies] - origin[:, None]
        root = data.root_link_pos_w - origin
        pose = data.joint_pos[:, env.joint_ids]
        capture = ~active | (age == 0)
        self.start_targets.copy_(torch.where(capture[:, None], entry_target, self.start_targets))
        self.entry_pose.copy_(torch.where(capture[:, None], pose, self.entry_pose))
        self.bias.copy_(torch.where(capture[:, None], (entry_target-pose).clamp(-0.8, 0.8), self.bias))
        self.start_root.copy_(torch.where(capture[:, None], root, self.start_root))
        self.start_sites.copy_(torch.where(capture[:, None, None], sites, self.start_sites))
        self.hand_anchors.copy_(torch.where(capture[:, None, None], hands, self.hand_anchors))
        self.bilateral_ticks.masked_fill_(capture, 0)
        self.recover_at.masked_fill_(capture, -1)
        duration, inset, clearance, advance, gain, landing = self.transfer_parameters.unbind(1)
        progress = (age.float() * env.step_dt / duration.clamp_min(0.15)).clamp(min=0)
        lift = self._smooth(progress * 2.8)
        reach = self._smooth((progress-0.18) / 0.66)
        settle = self._smooth((progress-0.71) / 0.29)
        blend = self._smooth(progress * 4)
        knee = sites[:, 6]
        first_knee = self.start_sites[:, 6]
        x0 = (first_knee-edge) @ direction
        goal = first_knee + ((inset-x0) * reach)[:, None] * direction
        goal[:, 2] = torch.lerp(first_knee[:, 2], top+clearance, lift) - (clearance-landing)*settle
        root_goal = self.start_root + (advance*self._smooth((progress-.1)/.8))[:, None]*direction
        root_goal[:, 2] += .035 * lift
        base_error = (root_goal-root).clamp(-.18,.18)
        joint_target = self.entry_pose.clone()
        left_cols, _, _ = self.limbs[0]
        joint_target[:, left_cols[3]] = torch.lerp(self.entry_pose[:, left_cols[3]], 2.30, lift)
        joint_target[:, left_cols[0]] -= 0.55 * lift
        joint_target[:, left_cols[4]] -= .18 * lift
        # A planted joint-space command inherits the actual landing preload. When the
        # pelvis advances, recalculate contact accommodation from the measured position.
        target = joint_target + self.bias
        jac = env.robot.root_physx_view.get_jacobians()
        body_reaction = -gain[:, None]*base_error + 56.*data.root_link_lin_vel_w
        body_reaction = body_reaction.clamp(-105,105)
        for limb, (cols, joints, dofs) in enumerate(self.limbs):
            if limb < 2:
                point = sites[:, 6+limb]
                body = env.site_ids[6+limb]
            else:
                point = hands[:, limb-2]
                body = self.hand_bodies[limb-2]
            j = point_jacobian(jac[:, body, :, dofs], point + origin-data.body_com_pos_w[:, body])
            if limb == 0:
                error = goal-knee
                force = torch.zeros_like(error)
            elif limb == 1:
                error = self.start_sites[:, 7]-point - .42*base_error
                force = .34 * body_reaction
            else:
                error = self.hand_anchors[:, limb-2]-point - .30*base_error
                force = .33 * body_reaction
            current = pose[:, cols]
            dq = (joint_target[:, cols]-current).unsqueeze(-1)
            correction, _ = torch.linalg.solve_ex(j @ j.mT+self.regularizer,
                                                   error.clamp(-.24,.24).unsqueeze(-1) - j @ dq)
            offset = (j.mT @ force.unsqueeze(-1)).squeeze(-1) / data.joint_stiffness[:, joints].clamp_min(1.)
            target[:, cols] += blend[:, None] * (
                (j.mT @ correction).squeeze(-1).clamp(-1.,1.) + offset.clamp(-.6,.6))
        target += .12 * residual.clamp(-1,1) * blend[:, None]
        target[:, env.wrist_joint] = env.fixed_wrist_target

        knee_load = env.last_metrics.get("knee_top_N")
        if knee_load is not None:
            both = active & (knee_load > 8).all(dim=1) & ((knee-edge) @ direction > 0.)
            self.bilateral_ticks.copy_(torch.where(both, self.bilateral_ticks+1, 0))
            ready = (self.bilateral_ticks >= 6) & (self.recover_at < 0)
            self.recover_at.copy_(torch.where(ready, age, self.recover_at))
        recover = active & (self.recover_at >= 0)
        target = self.recovery.desired(target, residual, recover, (age-self.recover_at).clamp_min(0))
        return torch.where(active[:, None], target, entry_target)
