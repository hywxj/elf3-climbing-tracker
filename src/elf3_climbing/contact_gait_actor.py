"""Initialize one contact-conditioned policy with the vendor's trainable proprioceptive layers."""

import xml.etree.ElementTree as ET

import torch
from torch import nn
from torch.nn import functional as F

from .contact_learning import CONTACT_OBS_DIM


class GaitContactActor(nn.Module):
    """All bodies and tasks use every layer. Loading a checkpoint needs no ONNX file."""

    def __init__(self):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(CONTACT_OBS_DIM))
        self.register_buffer("obs_scale", torch.ones(CONTACT_OBS_DIM))
        self.register_buffer("gait_mean", torch.zeros(96))
        self.register_buffer("gait_scale", torch.ones(96))
        for name in ("default", "nominal", "damping_offset"):
            self.register_buffer(name, torch.zeros(29))
        for name in ("action_scale", "kp_ratio"):
            self.register_buffer(name, torch.ones(29))
        self.encoder = nn.Sequential(
            nn.Linear(CONTACT_OBS_DIM, 192), nn.ELU(), nn.Linear(192, 128), nn.ELU()
        )
        self.planner = nn.Linear(128, 32)
        self.command = nn.Linear(160, 3)
        self.context = nn.Linear(160, 128)
        nn.init.zeros_(self.command.weight)
        nn.init.zeros_(self.command.bias)
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.context.bias)
        # Keep a fixed direction only as the initial forward-speed bias of a trainable layer.
        with torch.no_grad():
            self.command.bias[0] = 0.12
        sizes = (96, 512, 256, 128, 29)
        self.backbone = nn.ModuleList([nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])])

    @torch.no_grad()
    def initialize_vendor(self, names, nominal, drive_model):
        from .ground_locomotion import GroundLocomotion

        if list(names).index("l_wrist_y_joint") != 20:
            raise ValueError("Unexpected wrist index in ELF3 actuator order")
        gait = GroundLocomotion(names, "cpu")
        motors = ET.parse(drive_model).find("actuator")
        gains = {
            m.get("joint"): (float(m.get("kp")), float(m.get("kv")))
            for m in motors.iter("position")
        }
        kp, kd = torch.tensor([gains[n] for n in names]).T
        if (kp <= 0).any():
            raise ValueError("Invalid drive stiffness for gait conversion")
        self.nominal.copy_(torch.as_tensor(nominal))
        self.default.copy_(gait.default)
        self.gait_mean.copy_(gait.mean.flatten())
        self.gait_scale.copy_(gait.std.flatten())
        self.action_scale.copy_(gait.scale)
        self.kp_ratio.copy_(gait.kp / kp)
        self.damping_offset.copy_((kd - gait.kd) / kp)
        for layer, (weight, bias) in zip(self.backbone, gait.layers):
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)

    def action_and_plan(self, obs):
        state = self.encoder(((obs - self.obs_mean) / self.obs_scale).clamp(-8, 8))
        plan = self.planner(state).reshape(-1, 8, 4)
        logits, position = plan[:, :, 0], plan[:, :, 1:]
        valid = logits.sigmoid().unsqueeze(2)
        context = torch.cat((state, valid.flatten(1), (valid * position).flatten(1)), 1)
        return self.motor_target(obs, context), logits, position

    def motor_target(self, obs, context):
        q = obs[:, :29] + self.nominal
        velocity = 20 * obs[:, 29:58]
        previous = obs[:, 58:87] + self.nominal
        # Recover a vendor-scale previous action from the most recent actual drive command.
        # This also works at reset, with no policy-private recurrent state to synchronize.
        last = (
            q + (previous - q - self.damping_offset * velocity) / self.kp_ratio - self.default
        ) / self.action_scale
        centered = q - self.default
        centered = centered.clone()
        centered[:, 20] = q[:, 20] - previous[:, 20]
        joint_speed = velocity.clone()
        joint_speed[:, 20] = 0
        x = torch.cat(
            (
                4 * obs[:, 90:93],
                obs[:, 93:96],
                centered,
                joint_speed,
                last.clamp(-5, 5),
                self.command(context),
            ),
            1,
        )
        x = (x - self.gait_mean) / self.gait_scale
        for layer in self.backbone[:2]:
            x = F.elu(layer(x))
        x = F.elu(self.backbone[2](x) + self.context(context))
        desired = self.default + self.action_scale * self.backbone[3](x)
        target = q + self.kp_ratio * (desired - q) + self.damping_offset * velocity
        return target - self.nominal

    def forward(self, obs):
        return self.action_and_plan(obs)[0]

    def supervised_parameter_groups(self, lr):
        # Fine tune every copied layer, with a smaller update preserving proprioceptive feedback.
        core = list(self.backbone.parameters())
        ids = {id(x) for x in core}
        return [
            dict(params=core, lr=0.05 * lr),
            dict(params=[p for p in self.parameters() if id(p) not in ids], lr=lr),
        ]
