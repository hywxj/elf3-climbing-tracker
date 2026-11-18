"""Initialize newly varying flip inputs without inheriting constant-channel gains."""

import torch


def preserve_standing_feedback(policy, normalizer, wrist):
    """Train the wrist output row only; keep the learned body feedback function."""
    for parameter in policy.actor.parameters():
        parameter.requires_grad_(False)
    output = policy.actor[-1]
    output.weight.requires_grad_(True)
    output.bias.requires_grad_(True)
    weight_mask = torch.zeros_like(output.weight)
    bias_mask = torch.zeros_like(output.bias)
    weight_mask[wrist] = 1
    bias_mask[wrist] = 1
    handles = [
        output.weight.register_hook(lambda gradient: gradient * weight_mask),
        output.bias.register_hook(lambda gradient: gradient * bias_mask),
    ]
    # Frozen weights alone do not preserve feedback if their input scaling drifts.
    normalizer.until = int(normalizer.count)
    return handles


def standing_to_flip(policy, actor_normalizer, critic_normalizer, wrist, wrist_bias):
    # Five histories: angular velocity(3), gravity(3), q(29), qdot(29), action(29).
    channels = [h * 93 + offset + wrist for h in range(5) for offset in (6, 35, 64)]
    # Flip angle/requirement, gate allowed/settling, approach phase. Keep clearance.
    channels += [626, 627, 628, 629, 631]
    critic_channels = channels + [632 + wrist, 632 + 29 + wrist]
    if policy.actor[0].in_features != 632 or policy.critic[0].in_features != 697:
        raise ValueError("Standing transfer requires the 632/697 observation schema")
    with torch.no_grad():
        for network, normalizer, selected in (
            (policy.actor, actor_normalizer, channels),
            (policy.critic, critic_normalizer, critic_channels),
        ):
            # Preserve the learned contribution of all other channels. At the old
            # normalized mean these discarded channels contributed exactly zero.
            network[0].weight[:, selected] = 0
            normalizer._mean[:, selected] = 0
            normalizer._var[:, selected] = 1
            normalizer._std[:, selected] = 1
        # Seed this previously masked output from the flip reference end pose.
        # Runtime policy actions still use normal actuator targets/rate limits;
        # no reference trajectory or qpos is injected during rollout.
        policy.actor[-1].weight[wrist] = 0
        policy.actor[-1].bias[wrist] = wrist_bias
    return {
        "method": "standing_to_flip_reference_seed_v2",
        "actor_reinitialized_input_columns": channels,
        "critic_reinitialized_input_columns": critic_channels,
        "wrist_output_bias": float(wrist_bias),
        "other_output_rows_preserved": True,
        "scope": "Wrist goal seeded from reference, then PPO refinement. Any pre-PPO flip is initialization performance, not reward learning from scratch.",
    }
