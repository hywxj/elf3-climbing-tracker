"""Configuration parsing without importing Isaac Sim."""

import tomllib
from .runner import runner_config

RUN_FIELDS = {
    "num_envs",
    "iterations",
    "steps",
    "segment",
    "reward_stage",
    "knee_support_rule",
    "knee_wrist_contact",
    "seed",
    "no_curriculum",
    "flat_ground",
    "wrist_only",
    "approach_end_frame",
    "no_pose_init",
    "eval_interval",
    "early_stop_patience",
    "failure_warmup_iterations",
    "wrist_noise_std",
    "balance_noise_std",
}


def load_settings(path=None):
    if path is None:
        return {"run": {}, "ppo": {}}
    with open(path, "rb") as stream:
        data = tomllib.load(stream)
    if set(data) - {"run", "ppo"}:
        raise ValueError("Only [run] and [ppo] sections are supported")
    run, ppo = data.get("run", {}), data.get("ppo", {})
    if set(run) - RUN_FIELDS:
        raise ValueError(f"Unknown run settings: {set(run) - RUN_FIELDS}")
    for key in (
        "num_envs",
        "iterations",
        "steps",
        "eval_interval",
        "failure_warmup_iterations",
    ):
        if key in run and (type(run[key]) is not int or run[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    if "early_stop_patience" in run and (type(run["early_stop_patience"]) is not int or run["early_stop_patience"] < 0):
        raise ValueError("early_stop_patience must be a nonnegative integer")
    if "save_interval" in ppo and (type(ppo["save_interval"]) is not int or ppo["save_interval"] < 1):
        raise ValueError("save_interval must be a positive integer")
    for key, choices in {
        "segment": {"balance", "flip", "stand", "prepare", "approach", "walk", "knee_entry"},
        "reward_stage": {"prior", "refine"},
        "knee_support_rule": {"assisted-push", "push-off", "continuous"},
        "knee_wrist_contact": {"full", "distal-only", "tabletop", "tabletop-touch"},
    }.items():
        if key in run and run[key] not in choices:
            raise ValueError(f"Invalid {key}")
    if "approach_end_frame" in run and run["approach_end_frame"] not in (73, 128, 173):
        raise ValueError("approach_end_frame must be 73, 128 or 173")
    for key in ("no_curriculum", "no_pose_init", "flat_ground", "wrist_only"):
        if key in run and type(run[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if "wrist_noise_std" in run and not 0 < run["wrist_noise_std"] <= 0.02:
        raise ValueError("wrist_noise_std must be in (0, 0.02]")
    if "balance_noise_std" in run and not 0 < run["balance_noise_std"] <= 0.1:
        raise ValueError("balance_noise_std must be in (0, 0.1]")
    defaults = runner_config()
    if set(ppo) - set(defaults):
        raise ValueError(f"Unknown PPO settings: {set(ppo) - set(defaults)}")
    for key in ("policy", "algorithm"):
        if set(ppo.get(key, {})) - set(defaults[key]):
            raise ValueError(f"Unknown PPO {key} settings")
        if (
            ppo.get(key, {}).get("class_name", defaults[key]["class_name"])
            != defaults[key]["class_name"]
        ):
            raise ValueError("This tracker supports ActorCritic + PPO only")
    if ppo.get("logger", "tensorboard") != "tensorboard":
        raise ValueError("Only local tensorboard logging is enabled")
    return {"run": run, "ppo": ppo}
