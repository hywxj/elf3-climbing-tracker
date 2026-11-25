"""Rebuild the knee-entry and ground-approach teachers used to collect demonstrations."""

import json
from pathlib import Path

import torch

from .reference import digest


def source_inputs(path):
    request = json.loads((path.parent / "request.json").read_text())
    weights = torch.load(path, map_location="cpu", weights_only=True)
    contract = weights["infos"]["contract"]
    if (
        contract["segment"] != "knee_entry"
        or contract["task"]["action_prior"] != "v4_contact_guided_transfer"
        or request["segment"] != "knee_entry"
    ):
        raise ValueError("Entry weights must come from contact-guided knee-entry training")
    files = {
        key: Path(request[key]).expanduser().resolve()
        for key in ("reference", "knee_transfer_guide", "preload_example")
    }
    hashes = {name + "_sha256": digest(filename) for name, filename in files.items()}
    if hashes["knee_transfer_guide_sha256"] != contract["task"]["transfer_guide_sha256"]:
        raise ValueError("The transfer guide differs from the file used for the entry checkpoint")
    hashes["entry_checkpoint_sha256"] = digest(path)
    return request, weights, files, hashes


def knee_entry_config(request, files, *, seed, num_envs, device):
    """Restore the teacher's contact rules and assets after launching the Isaac app."""
    from .environment import TrackerCfg

    cfg = TrackerCfg()
    cfg.seed, cfg.scene.num_envs, cfg.sim.device = seed, num_envs, device
    cfg.segment = "knee_entry"
    cfg.reference_path = str(files["reference"])
    cfg.knee_transfer_guide = str(files["knee_transfer_guide"])
    cfg.preload_example = str(files["preload_example"])
    cfg.knee_support_rule = request["knee_support_rule"]
    cfg.knee_wrist_contact = request["knee_wrist_contact"]
    cfg.random_start = cfg.curriculum_enabled = False
    return cfg


def _normalized_actor(weights, obs_dim, critic_dim, policy_options, device):
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization

    model = ActorCritic(obs_dim, critic_dim, 29, **policy_options)
    model.load_state_dict(weights["model_state_dict"])
    normalizer = EmpiricalNormalization(shape=[obs_dim])
    normalizer.load_state_dict(weights["obs_norm_state_dict"])
    return torch.nn.Sequential(normalizer, model.actor).to(device).eval()


def frozen_entry(weights, ppo_file, device):
    tensors = weights["model_state_dict"]
    options = json.loads(ppo_file.read_text())["policy"]
    if options.pop("class_name") != "ActorCritic":
        raise ValueError("Expected a knee-entry ActorCritic checkpoint")
    actor = _normalized_actor(
        weights,
        tensors["actor.0.weight"].shape[1],
        tensors["critic.0.weight"].shape[1],
        options,
        device,
    )
    return actor.requires_grad_(False)


def ground_teacher_actor(weights, cfg, device):
    options = dict(
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
        noise_std_type="log",
    )
    return _normalized_actor(weights, cfg.observation_space, cfg.state_space, options, device)
