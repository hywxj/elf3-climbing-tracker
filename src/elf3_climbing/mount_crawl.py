"""Contact-triggered crawl from a loaded knee to the two feet on a known tabletop."""

import torch


class MountCrawl:
    """Coordinate support, torso motion and swinging limbs in one damped Jacobian solve."""

    parameter_names = (
        "tempo",
        "body_advance_m",
        "body_lift_m",
        "swing_height_m",
        "pull_N",
        "wrist_anchor_weight",
    )

    def __init__(self, env):
        self.env = env
        n, dev = env.num_envs, env.device
        self.parameters = torch.tensor([1.77, 0.19, 0.044, 0.195, 80.45, 0.49], device=dev).repeat(
            n, 1
        )
        self.phase = torch.zeros_like(env.starts)
        self.clock = torch.zeros_like(self.phase)
        self.held = torch.zeros_like(self.phase)
        self.root0 = torch.zeros(n, 3, device=dev)
        self.xyz0 = torch.zeros(n, 6, 3, device=dev)
        self.up0 = torch.zeros(n, 3, device=dev)
        self.q0 = torch.zeros(n, 29, device=dev)
        self.bias = torch.zeros_like(self.q0)
        self.fwd = env.knee_contacts.trace_inward
        self.side = torch.stack((-self.fwd[1], self.fwd[0], self.fwd[2]))
        self.wp = env.robot.find_bodies(["l_wrist_y_link", "r_wrist_y_link"], preserve_order=True)[
            0
        ]
        self.bodies = [*env.site_ids[:2], *env.site_ids[6:8], *self.wp]
        self.dofs = [6 + j for j in env.joint_ids]
        self.identity = torch.eye(29, device=dev)
        self.last_q = env.ref["joint_pos"][-1]
        self.joint_names = env.reference_np["joint_names"].tolist()
        self.knees = [self.joint_names.index(side + "_knee_y_joint") for side in ("l", "r")]

    def desired(self, entry_target, residual, active, age):
        e = self.env
        d = e.robot.data
        origin = e.scene.env_origins
        sites = e.site_positions()
        xyz = (
            torch.cat((sites[:, :2], sites[:, 6:8], d.body_link_pos_w[:, self.wp]), 1)
            - origin[:, None]
        )
        root = d.root_link_pos_w - origin
        top = e.knee_contacts.top_heights - origin[:, 2]
        q = d.joint_pos[:, e.joint_ids]
        reset = ~active | (age == 0)
        self.root0.copy_(torch.where(reset[:, None], root, self.root0))
        self.xyz0.copy_(torch.where(reset[:, None, None], xyz, self.xyz0))
        self.q0.copy_(torch.where(reset[:, None], q, self.q0))
        self.bias.copy_(
            torch.where(reset[:, None], (entry_target - q).clamp(-0.45, 0.45), self.bias)
        )
        self.phase.masked_fill_(reset, 0)
        self.held.masked_fill_(reset, 0)
        self.clock.masked_fill_(reset, 0)

        tempo, forward, lift, arc, pull, hand_weight = self.parameters.unbind(1)
        self.clock += active.long()
        u = (self.clock * e.step_dt * tempo / 1.40).clamp(0, 1)
        u = u.square() * (3 - 2 * u)
        k = e.knee_contacts
        left_knee = k.surface_loads(k.knee_names[0], sites[:, 6])[0]
        feet = torch.stack([k.surface_loads(name)[0] for name in k.foot_names], 1)
        contact = torch.where(
            self.phase == 0,
            left_knee > 12,
            torch.where(self.phase == 1, feet[:, 1] > 20, feet[:, 0] > 20),
        )
        self.held.copy_(torch.where(contact & active & (u > 0.5), self.held + 1, 0))
        next_phase = (self.held >= 4) & (self.phase < 3)
        self.phase += next_phase.long()
        self.held.masked_fill_(next_phase, 0)
        self.clock.masked_fill_(next_phase, 0)
        u = torch.where(next_phase, 0.0, u)
        knee, right, left, stand = [self.phase == phase for phase in range(4)]
        # Goals for existing supports are measured at handoff or contact acquisition.
        self.xyz0.copy_(torch.where(next_phase[:, None, None], xyz, self.xyz0))
        self.root0.copy_(torch.where(next_phase[:, None], root, self.root0))
        start = self.xyz0
        goal = start.clone()
        w = torch.zeros_like(xyz[..., 0])
        w[:, 4:] = hand_weight[:, None] * (1 - stand.float() * u)[:, None]
        w[:, 3] = torch.where(knee, 1.1, 0.0)
        w[:, 2] = torch.where(knee, 0.95, torch.where(right, 1.1, 0.0))
        w[:, 1] = torch.where(right | left | stand, 1.15, 0.0)
        w[:, 0] = torch.where(knee, 0.40, torch.where(left | stand, 1.15, 0.0))

        def swing(slot, mask, inset, finish_height, clearance):
            initial = start[:, slot]
            final = initial + (inset - (initial - k.trace_edge) @ self.fwd)[:, None] * self.fwd
            final[:, 2] = top + finish_height
            t = u[:, None]
            value = initial + (final - initial) * t
            value[:, 2] += torch.sin(torch.pi * u) * clearance
            goal[:, slot] = torch.where(mask[:, None], value, goal[:, slot])

        swing(2, knee, 0.15, 0.055, arc)
        # While kneeling, fold the loose shin and bring the sole above the edge.
        swing(0, knee, -0.13, 0.065, arc * 0.35)
        swing(1, right, 0.22, 0.002, arc + 0.07)
        swing(0, left, 0.25, 0.002, arc + 0.07)
        root_goal = (
            self.root0
            + (u * torch.where(knee, forward, torch.where(right | left, 0.08, 0.0)))[:, None]
            * self.fwd
        )
        # Shift toward the support opposite the swinging foot.
        root_goal += (u * torch.where(right, 0.075, torch.where(left, -0.055, -0.02)))[
            :, None
        ] * self.side
        root_goal[:, 2] += u * torch.where(
            knee,
            lift,
            torch.where(right, 0.13, torch.where(left, 0.12, (top + 0.99 - self.root0[:, 2]))),
        )
        root_shift = (0.60 * (root_goal - root) - 0.045 * d.root_link_lin_vel_w).clamp(-0.12, 0.12)
        # Tilt the torso toward upright only after first securing the opposite knee.
        qbase = d.root_link_quat_w
        qw, qx, qy, qz = qbase.unbind(1)
        z = torch.stack(
            (2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)), 1
        )
        vertical = torch.zeros_like(z)
        vertical[:, 2] = 1.0
        self.up0.copy_(torch.where(reset[:, None], z, self.up0))
        up_goal = torch.lerp(self.up0, vertical, ((~knee) * (0.8 + 0.2 * stand) * u)[:, None])
        rot = 0.65 * torch.cross(z, up_goal, dim=1) - 0.055 * d.root_link_ang_vel_w
        error = (
            goal
            - xyz
            - root_shift[:, None]
            - torch.cross(rot[:, None].expand_as(xyz), xyz - root[:, None], dim=2)
        )
        j_body = e.robot.root_physx_view.get_jacobians()[:, self.bodies][:, :, :, self.dofs]
        offset = xyz + origin[:, None] - d.body_com_pos_w[:, self.bodies]
        j = j_body[:, :, :3] + torch.cross(
            j_body[:, :, 3:], offset[:, :, :, None].expand_as(j_body[:, :, 3:]), dim=2
        )
        weight = w[:, :, None, None]
        a = (j * weight).flatten(1, 2)
        b = (error.clamp(-0.18, 0.18) * w[:, :, None]).flatten(1)
        # The regularizer retains the current loaded stance, then favors extension when both soles bear.
        posture = torch.lerp(self.q0, self.last_q.expand_as(q), (stand.float() * u)[:, None])
        posture[:, self.knees[0]] += knee * u * (2.4 - self.q0[:, self.knees[0]])
        posture[:, 3] += knee * u * (-2.4 - self.q0[:, 3])
        posture[:, 5] += knee * u * (-0.1 - self.q0[:, 5])
        delta = 0.85 * (posture - q)
        damping = 0.10**2
        x, _ = torch.linalg.solve_ex(
            a.mT @ a + damping * self.identity, a.mT @ b[:, :, None] + damping * delta[:, :, None]
        )
        result = q + x.squeeze(2).clamp(-0.85, 0.85) + self.bias * (1 - stand.float() * u)[:, None]
        # Pull the body forward by commanding backward force at the anchored wrists.
        force = -(pull * u * ~stand)[:, None] * self.fwd
        traction = (j[:, 4:].transpose(2, 3) @ force[:, None, :, None]).sum(1).squeeze(-1)
        result += (traction / d.joint_stiffness[:, e.joint_ids].clamp_min(1.0)).clamp(-0.45, 0.45)
        result += 0.15 * residual.clamp(-1, 1)
        result[:, e.wrist_joint] = e.fixed_wrist_target
        mix = (age.float() / 8).clamp(0, 1)[:, None]
        return torch.where(active[:, None], torch.lerp(entry_target, result, mix), entry_target)
