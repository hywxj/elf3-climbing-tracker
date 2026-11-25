"""Phase-conditioned shaping for the 0.70 m elbow-supported knee-entry task."""

import torch

from .isaac_knee_entry import PLANTED_SLIP_M_S


class KneeEntryReward:
    """Bound milestone credit per episode and pay for the physical support sequence."""

    def __init__(self, env):
        self.env = env
        self.version = ("knee_entry_surface_contact_v4" if env.cfg.knee_wrist_contact == "full"
                        else "knee_entry_surface_contact_v3")
        self.phased = env.cfg.knee_support_rule != "continuous"
        self.assisted = env.cfg.knee_support_rule == "assisted-push"
        if self.phased:
            self.version = f"knee_entry_planted_push_v5_{env.cfg.knee_wrist_contact}"
        if self.assisted:
            version = "contact_guided_v8" if getattr(env.cfg, "knee_transfer_guide", None) else "assisted_transfer_v7"
            self.version = f"knee_entry_{version}_{env.cfg.knee_wrist_contact}"
        self.hold = torch.zeros_like(env.starts)
        self.endpoint_hold = self.hold.clone()
        self.best_hold = torch.zeros_like(env.starts, dtype=torch.float32)
        self.best_settled_hold = self.best_hold.clone()
        self.best_reach = self.best_hold.clone()
        self.best_assist = self.best_hold.clone()
        self.credited_takeoff = torch.zeros_like(env.starts, dtype=torch.bool)

    @staticmethod
    def reach(distance):
        return torch.exp(-distance / 0.18)

    def reset(self, ids):
        e = self.env
        target = e.ref["site_pos"][e.segment_end, 7]
        distance = (e.ref["site_pos"][e.starts[ids], 7] - target).norm(dim=1)
        self.best_reach[ids] = self.reach(distance)
        self.best_assist[ids] = 0
        self.best_hold[ids] = 0
        self.best_settled_hold[ids] = 0
        self.hold[ids] = 0
        self.endpoint_hold[ids] = 0
        self.credited_takeoff[ids] = False

    def compute(self, frames, pos_error, angle_error, joint_mse):
        e = self.env
        m = e.last_metrics
        dt = e.step_dt
        grounded_phase = frames < e.knee_contacts.takeoff
        moving = frames < e.segment_end
        wrist_forbidden = m.get("wrist_forbidden", m["wrist_loaded"])
        clean = ~wrist_forbidden & ~m["terminated"] & e.gate_allowed
        arm = (m["forearm_top_load_N"].amin(1) / 12).clamp(0, 1)
        feet = (m["foot_ground_load_N"].amin(1) / 16).clamp(0, 1)
        foot_release = (1 - m["max_foot_load_N"] / 16).clamp(0, 1)
        clearance = (m["min_foot_clearance_m"] / 0.04).clamp(0, 1)
        posture = torch.exp(-joint_mse / 0.15 - angle_error.square() / 0.36)
        if e.cfg.reward_stage == "refine":
            posture = 0.35 * posture

        # Track the demo's downward preload and upward push only with measured support.
        vertical = e.robot.data.root_link_lin_vel_w[:, 2]
        desired = e.ref["root_lin_vel"][frames, 2]
        vz_tracking = torch.exp(-((vertical - desired) / 0.15).square())
        dz = (
            e.robot.data.root_link_pos_w[:, 2]
            - e.scene.env_origins[:, 2]
            - e.ref["root_pos"][frames, 2]
        )
        preload = torch.exp(-(dz / 0.045).square())
        alignment = torch.exp(-0.5 * (m["forearm_target_error_m"] / 0.06).square())
        if self.phased:
            pre_lift = ~(m["takeoff_observed"] | m["takeoff_assumed"]) & moving
            loaded = m["foot_ground_patch_N"] > 8.0
            slip = torch.where(loaded, m["foot_ground_slip_m_s"], 0.0)
            planted = torch.exp(-(slip.amax(1) / PLANTED_SLIP_M_S).square())
            foot_support = (m["weaker_foot_bodyweight"] / 0.15).clamp(0, 1) * planted
            extending = frames >= e.knee_contacts.push_start
            arm_load = m.get("arm_support_top_N", m["forearm_top_load_N"])
            arm_quality = alignment * (arm_load / 12).clamp(0, 1)
            if self.assisted:
                # Allow slight timing error in the push, then stop asking for a ground support.
                stance = pre_lift & (frames <= e.knee_contacts.takeoff + 2)
                feet_quality = (m["foot_ground_patch_N"] / 40).clamp(0, 1) * torch.exp(
                    -(slip / PLANTED_SLIP_M_S).square())
                # Provide a bounded learning signal even when the opposite foot or arm is missing.
                limbs = torch.cat((arm_quality, feet_quality), dim=1)
                dwell = (m["support_streak"] * dt / 0.10).clamp(0, 1)
                acquire = (0.25 * limbs.mean(1) + 0.75 * limbs.amin(1) * dwell) * stance * clean
                simultaneous = arm_quality.amin(1) * foot_support
                upward = (torch.minimum(m["com_vz_m_s"], m["hip_vz_m_s"]) / 0.35).clamp(0, 1)
                forward = (m["hip_forward_m_s"] / 0.25).clamp(0, 1)
                movement = torch.minimum(upward, forward)
                # Keep the elbows loaded during extension. Following release, arms can unload.
                ground_rate = stance * clean * simultaneous * torch.where(
                    extending, 5.0 * movement, 0.6 * preload + 0.6 * vz_tracking + 0.2 * posture)
                # Losing a single limb before the push must not be cheaper than holding a planted stance.
                support_loss = (1 - limbs.amin(1)) * stance * ~m["push_release_window"]
            else:
                acquire = arm_quality.amin(1) * foot_support * clean * pre_lift
                rise = (m["com_vz_m_s"] / 0.35).clamp(0, 1)
                preload_rate = (~extending) * (0.5 * preload + 0.5 * vz_tracking + 0.2 * posture)
                push_rate = extending * m["forearm_assist_recent"] * 2.0 * rise
                ground_rate = pre_lift * clean * foot_support * (preload_rate + push_rate)
            assist_gain = (acquire - self.best_assist).clamp_min(0)
            self.best_assist += assist_gain
            # Keep the measured 0.3–3 m/s slips distinguishable even far beyond the plant threshold.
            slip_cost = torch.log1p((slip / PLANTED_SLIP_M_S).square()).sum(1)
        else:
            arm_acquisition = (
                alignment * (0.25 + 0.75 * (m["forearm_top_load_N"] / 12).clamp(0, 1))
            ).mean(1)
            ground_rate = (
                grounded_phase
                * clean
                * (0.3 * arm + feet * (
                    0.4 * arm_acquisition + arm * (0.7 * preload + 0.7 * vz_tracking + 0.3 * posture)
                ))
            )
        # While the reference moves, provide partial credit for unloading and tucking.
        # Once the reference ends, waiting on one foot cannot earn tracking or survival pay.
        distance = m["right_knee_goal_distance_m"]
        if self.phased:
            valid_flight = clean & e.knee_contacts.flight
            carry = torch.ones_like(distance)
            if self.assisted:
                valid_flight = valid_flight & ~m["transfer_lost"]
                carry = torch.where(m["knee_bearing"], 1.0, m["arm_carry_fraction"])
            air_rate = carry * (valid_flight & moving) * foot_release * clearance * (
                0.8 + self.reach(distance) + 0.3 * posture)
        else:
            valid_flight = clean & ~grounded_phase & e.knee_contacts.flight & (arm >= 0.25)
            air_rate = (~grounded_phase & moving & clean) * (
                0.3 * arm
                + arm * foot_release * clearance * (0.8 + self.reach(distance) + 0.3 * posture)
            )
        cost_rate = (
            0.25
            + 0.5 * (~grounded_phase) * (1 - foot_release)
            + wrist_forbidden.float()
            + 0.02 * (e.actions - e.previous_actions).square().mean(1)
        )

        # CSI may start after liftoff. Only a measured first release earns this bonus.
        first_lift = (
            m["takeoff_observed"]
            & e.knee_contacts.flight
            & clean
            & ~self.credited_takeoff
            & (m["min_foot_clearance_m"] > 0.01)
        )
        self.credited_takeoff |= first_lift
        reach = self.reach(distance) * valid_flight * foot_release * clearance
        if self.assisted:
            reach = reach * carry
        reach_gain = (reach - self.best_reach).clamp_min(0)
        self.best_reach += reach_gain

        supported = clean & m["right_knee_supported"]
        self.hold.copy_(torch.where(supported, self.hold + 1, 0))
        duration = (self.hold.float() / e.required_hold_steps).clamp(max=1)
        hold_gain = (duration - self.best_hold).clamp_min(0)
        self.best_hold += hold_gain

        # Grade the distance to the existing terminal stability limits while loaded.
        # The original hard gate still determines success and curriculum promotion.
        error_ratio = torch.stack(
            [
                pos_error / 0.15,
                angle_error / 0.4,
                joint_mse / 0.12,
                m["root_velocity_error_m_s"] / 0.25,
                m["root_angular_velocity_error_rad_s"] / 0.5,
            ]
        ).amax(0)
        stability = torch.exp(-(error_ratio - 1).clamp_min(0))
        self.endpoint_hold.copy_(torch.where(
            supported & ~moving & m["gripper_ready"], self.endpoint_hold + 1, 0
        ))
        endpoint_duration = (self.endpoint_hold.float() / e.required_hold_steps).clamp(max=1)
        stable_duration = (m["stable_hold_steps"].float() / e.required_hold_steps).clamp(max=1)
        settled = 0.5 * (endpoint_duration * stability + stable_duration)
        settled_gain = (settled - self.best_settled_hold).clamp_min(0)
        self.best_settled_hold += settled_gain
        success = m["success"]
        terms = {
            "ground_tracking": dt * ground_rate,
            "air_tracking": dt * air_rate,
            "takeoff": first_lift.float(),
            "knee_approach": 2 * reach_gain,
            "supported_hold": 2 * hold_gain,
            "settled_hold": 3 * settled_gain,
            "cost": -dt * cost_rate,
            "terminal": 8 * success.float()
            - 3 * m["terminated"].float()
            - (m["expired"] & ~success & ~m["terminated"]).float(),
        }
        if self.phased:
            terms["forearm_acquisition"] = 0.5 * assist_gain
            terms["stance_slip"] = -0.8 * dt * slip_cost
        if self.assisted:
            terms["early_support_loss"] = -dt * support_loss
            transferring = (m["push_started"] | e.knee_contacts.flight) & ~m["knee_bearing"]
            terms["unsupported_transfer"] = -2 * dt * transferring * (1 - carry)
            # Coach the transition while the last foot still carries load. A binary
            # flight gate alone supplies no progress signal to a one-foot stance.
            unloading = torch.exp(-m["max_foot_load_N"].clamp_min(0) / 55.0)
            releasing = (m["push_started"] & ~m["takeoff_observed"] & moving & clean
                         & (frames >= e.knee_contacts.takeoff - 2) & ~m["transfer_lost"])
            terms["assisted_release"] = 3 * dt * releasing * carry * unloading * (
                0.5 * movement + 0.5 * self.reach(distance))
        reward = torch.stack(list(terms.values())).sum(0)
        e.extras["log"] = {f"KneeReward/{k}": v.mean() for k, v in terms.items()}
        e.extras["log"].update(
            {
                "Knee/right_top_load_N": m["right_knee_top_load_N"].mean(),
                "Knee/both_forearms_top_load_min_N": m["forearm_top_load_N"].amin(1).mean(),
                "Knee/supported_fraction": supported.float().mean(),
                "Knee/takeoff_observed_fraction": m["takeoff_observed"].float().mean(),
                "Knee/mean_best_hold_s": (self.best_hold * e.required_hold_steps * dt).mean(),
                "Episodes/success_fraction": e.outcome_counts[1].float()
                / e.outcome_counts[0].clamp(min=1),
                "Curriculum/initial_phase": torch.tensor(e.curriculum.phase, device=e.device),
            }
        )
        if self.phased:
            e.extras["log"]["Knee/mean_loaded_foot_slip_m_s"] = slip.sum() / loaded.sum().clamp_min(1)
            e.extras["log"]["Knee/planted_push_fraction"] = m["bilateral_push"].float().mean()
        if self.assisted:
            e.extras["log"]["Knee/simultaneous_support_fraction"] = m["simultaneous_support"].float().mean()
        return reward
