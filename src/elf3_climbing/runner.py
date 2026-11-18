"""Compatibility adapter and PPO configuration for the pinned local RSL-RL."""


class LegacyRslAdapter:
    """Adapt DirectRLEnv dict observations to the existing local RSL-RL tuple API."""

    def __init__(self, env):
        self.env = env
        self.unwrapped = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_actions = 29
        self.max_episode_length = env.max_episode_length
        self.cfg = env.cfg
        self.obs, _ = env.reset()
        self.initial_normalizers = None

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def get_observations(self):
        actor, critic = self.obs["policy"], self.obs["critic"]
        if self.initial_normalizers is not None:
            actor = self.initial_normalizers[0](actor)
            critic = self.initial_normalizers[1](critic)
        return actor, {"observations": {"critic": critic}}

    def step(self, actions):
        self.obs, rewards, terminated, truncated, extras = self.env.step(actions)
        extras["observations"] = {"critic": self.obs["critic"]}
        # These episodes end at a finite reference boundary, not an infinite-horizon timeout.
        return self.obs["policy"], rewards, (terminated | truncated).long(), extras


def runner_config(overrides=None):
    cfg = {
        "num_steps_per_env": 48,
        "save_interval": 20,
        "empirical_normalization": True,
        "logger": "tensorboard",
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [256, 128, 128],
            "critic_hidden_dims": [256, 128, 128],
            "activation": "elu",
            "init_noise_std": 0.01,
            "noise_std_type": "log",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.1,
            "entropy_coef": 0.0,
            "num_learning_epochs": 2,
            "num_mini_batches": 4,
            "learning_rate": 1e-4,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }

    for key, value in (overrides or {}).items():
        if isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg
