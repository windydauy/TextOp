from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

LOW_FREQ_SCALE = 0.5


def _ppo_cfg(*, model_class: str = "MLPModel", algorithm_class: str = "PPO") -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name=model_class,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name=model_class,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            class_name=algorithm_class,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        num_steps_per_env=24,
        max_iterations=100000,
        save_interval=500,
        experiment_name="g1_flat",
    )


def G1FlatPPORunnerCfg() -> RslRlOnPolicyRunnerCfg:
    return _ppo_cfg()


def G1FlatPPOModNormRunnerCfg() -> RslRlOnPolicyRunnerCfg:
    return _ppo_cfg(model_class="ActorCriticMNMLP", algorithm_class="PPO_MNMLP")


def G1FlatLowFreqPPORunnerCfg() -> RslRlOnPolicyRunnerCfg:
    cfg = G1FlatPPORunnerCfg()
    cfg.num_steps_per_env = round(cfg.num_steps_per_env * LOW_FREQ_SCALE)
    cfg.algorithm.gamma = cfg.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
    cfg.algorithm.lam = cfg.algorithm.lam ** (1 / LOW_FREQ_SCALE)
    return cfg
