"""Configuration parsing without importing Isaac Sim."""

import tomllib
from .runner import runner_config

RUN_FIELDS = {
    "num_envs",
    "iterations",
    "steps",
    "segment",
    "reward_stage",
    "seed",
    "no_curriculum",
    "no_pose_init",
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
    for key in ("num_envs", "iterations", "steps"):
        if key in run and (type(run[key]) is not int or run[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    for key, choices in {
        "segment": {"flip", "stand", "prepare"},
        "reward_stage": {"prior", "refine"},
    }.items():
        if key in run and run[key] not in choices:
            raise ValueError(f"Invalid {key}")
    for key in ("no_curriculum", "no_pose_init"):
        if key in run and type(run[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
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
