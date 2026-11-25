"""Actuated contact transfer for the 0.70 m right-knee entry."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .guided_knee import GuidedKneeTargets


def point_jacobian(jacobian, offset):
    """Translate a link COM Jacobian to a point on the same link."""
    return jacobian[:, :3] + torch.cross(
        jacobian[:, 3:], offset.unsqueeze(-1).expand_as(jacobian[:, 3:]), dim=1
    )


class KneeTransferTargets(GuidedKneeTargets):
    """Use a physics-tested leg trajectory and control the arms at the table."""

    def __init__(self, env, preload_example, guide):
        super().__init__(env, preload_example)
        path = Path(guide)
        n = env.segment_end - env.segment_start + 1
        expected = env.reference_manifest["pack_sha256"]
        shapes = {
            "leg_targets_rad": (n, 15),
            "arm_blend": (n,),
            "wrench_ramp": (n,),
            "arm_anchor_shift_m": (2,),
            "arm_depth_m": (2,),
            "normal_load_N": (2,),
            "pull_N": (),
            "landing_scale": (3,),
            "landing_pull_gains": (4,),
            "landing_root_pose": (7,),
            "landing_joint_pos": (29,),
        }
        with np.load(path, allow_pickle=False) as a:
            if (
                str(a["schema"]) != "elf3_knee_transfer_v1"
                or str(a["reference_sha256"]) != expected
            ):
                raise ValueError("Knee transfer guide must match the reference pack")
            self.lift_gate_frame = int(a["lift_gate_frame"])
            if not (env.knee_contacts.push_start < self.lift_gate_frame < env.segment_end):
                raise ValueError("Transfer guide has an invalid lift gate frame")
            for name, shape in shapes.items():
                value = a[name]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid transfer guide array: {name}")
                setattr(self, name, torch.as_tensor(value, dtype=torch.float32, device=env.device))
            if not np.isclose(np.linalg.norm(a["landing_root_pose"][3:]), 1.0, atol=1e-4):
                raise ValueError("Transfer guide has an invalid landing quaternion")
        self.anchor = env.ref["site_pos"][env.knee_contacts.takeoff - 4, 2:6].reshape(2, 2, 3)
        self.weights = torch.tensor([0.75, 0.75, 1.0, 0.85], device=env.device)
        self.regularization = 0.04**2 * torch.eye(4, device=env.device)
        self.touchdown_step = torch.full_like(env.starts, -1)
        env.task_report.update(
            action_prior="v4_contact_guided_transfer",
            transfer_guide=str(path.resolve()),
            transfer_guide_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reference_lift_gate_frame=self.lift_gate_frame,
            terminal_pose="physics roll-out knee hold",
        )

    def reset(self, env_ids):
        self.touchdown_step[env_ids] = -1

    def desired(self, residual):
        e = self.env
        k = e.knee_contacts
        row = e.frames() - e.segment_start
        jacobians = e.robot.root_physx_view.get_jacobians()
        sites = e.site_positions()
        data = e.robot.data
        target = self.preloaded_targets(residual, jacobians, sites)
        target[:, :15] = self.leg_targets_rad[row] + 0.5 * residual[:, :15].clamp(-2.0, 2.0)

        new_touchdown = (k.knee_load_steps >= 2) & (self.touchdown_step < 0)
        self.touchdown_step.copy_(torch.where(new_touchdown, e.ref_steps, self.touchdown_step))
        landed = ((e.ref_steps - self.touchdown_step) / 8).clamp(0, 1) * (self.touchdown_step >= 0)
        normal_scale, anchor_return, pull_scale = self.landing_scale.unbind(0)

        forward = torch.nn.functional.pad(e.advance_direction, (0, 1))
        lateral = torch.stack((-forward[1], forward[0], forward[2]))
        shift = self.arm_anchor_shift_m[0] * forward + self.arm_anchor_shift_m[1] * lateral
        origin = e.scene.env_origins
        root_error = e.ref["root_pos"][e.segment_end] + origin - data.root_link_pos_w
        speed = data.root_link_lin_vel_w
        kp_x, kd_x, kp_y, kd_y = self.landing_pull_gains.unbind()
        reaction_x = (kp_x * (root_error @ forward) - kd_x * (speed @ forward)).clamp(-125, 125)
        reaction_y = (kp_y * (root_error @ lateral) - kd_y * (speed @ lateral)).clamp(-105, 105)
        depth = self.arm_depth_m * (1 - 0.25 * landed[:, None])
        for side, (body, cols, joints, dofs) in enumerate(self.arms):
            jac = jacobians[:, body, :, dofs]
            com = data.body_com_pos_w[:, body]
            forearm, elbow = sites[:, 2 + side], sites[:, 4 + side]
            forearm_j = point_jacobian(jac, forearm - com)
            elbow_j = point_jacobian(jac, elbow - com)
            goal = self.anchor[0, side] + origin + (1 - anchor_return * landed[:, None]) * shift
            goal[:, 2] -= depth[:, 0]
            elbow_error = self.anchor[1, side, 2] + origin[:, 2] - depth[:, 1] - elbow[:, 2]
            error = torch.cat((goal - forearm, elbow_error[:, None]), dim=1)
            j = torch.cat((forearm_j, elbow_j[:, 2:3]), dim=1) * self.weights[None, :, None]
            current = data.joint_pos[:, joints]
            dq = (target[:, cols] - current).unsqueeze(2)
            correction, _ = torch.linalg.solve_ex(
                j @ j.transpose(1, 2) + self.regularization,
                (error * self.weights).unsqueeze(2) - j @ dq,
            )
            arm = current + (dq + j.transpose(1, 2) @ correction).squeeze(2).clamp(-0.65, 0.65)
            arm = torch.lerp(target[:, cols], arm, self.arm_blend[row, None])

            # Keep both support targets during knee settling. Taper the pull and
            # the extra downward command after the knee carries measured load.
            pull = self.pull_N * (1 - landed * (1 - pull_scale)) - landed * reaction_x
            force = pull[:, None] * forward - (landed * reaction_y)[:, None] * lateral
            force[:, 2] = -self.normal_load_N[side] * (1 - landed * (1 - normal_scale))
            offset = (forearm_j.transpose(1, 2) @ force.unsqueeze(2)).squeeze(2)
            offset = offset / data.joint_stiffness[:, joints].clamp_min(1.0)
            target[:, cols] = arm + self.wrench_ramp[row, None] * offset.clamp(-0.85, 0.85)
        return target
