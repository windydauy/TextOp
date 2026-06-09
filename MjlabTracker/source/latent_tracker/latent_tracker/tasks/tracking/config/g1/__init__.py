from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from . import flat_env_cfg
from .agents import rsl_rl_ppo_cfg


def _register(task_id: str, env_factory, rl_factory) -> None:
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_factory(),
        play_env_cfg=env_factory(play=True),
        rl_cfg=rl_factory(),
        runner_cls=MotionTrackingOnPolicyRunner,
    )


_register(
    "Mjlab-LatentTracker-Flat-G1",
    flat_env_cfg.G1FlatEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-PrivPrivObs",
    flat_env_cfg.G1FlatPrivPrivObsEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-PropPropObs",
    flat_env_cfg.G1FlatPropPropObsEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-NoisePrivObs",
    flat_env_cfg.G1FlatNoisePrivObsEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-ProjGravObs",
    flat_env_cfg.G1FlatProjGravObsEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
# _register(
#     "Mjlab-LatentTracker-Flat-G1-ProjGravObs-MNMLP",
#     flat_env_cfg.G1FlatProjGravObsEnvCfg,
#     rsl_rl_ppo_cfg.G1FlatPPOModNormRunnerCfg,
# )
_register(
    "Mjlab-LatentTracker-Flat-G1-ProjGravObs-MNMLP-LargeHand",
    flat_env_cfg.G1FlatProjGravObsEnvCfg_LargeHand,
    rsl_rl_ppo_cfg.G1FlatPPOModNormRunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-ProjGravObs-MNMLP-LargeHandHeavy",
    flat_env_cfg.G1FlatProjGravObsEnvCfg_LargeHandHeavy,
    rsl_rl_ppo_cfg.G1FlatPPOModNormRunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-Wo-State-Estimation",
    flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-ProjGravAnchorObs",
    flat_env_cfg.G1FlatProjGravAnchorObsEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-ProjGravAnchorObs-MotionAE",
    flat_env_cfg.G1FlatProjGravAnchorObsMotionAEEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-ProjGravAnchorEEObs",
    flat_env_cfg.G1FlatProjGravAnchorEEObsEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
# _register(
#     "Mjlab-LatentTracker-Flat-G1-ProjGravAnchorObs-TransformerVAE-NMMLP",
#     flat_env_cfg.G1FlatProjGravAnchorObsTransformerVAEEnvCfg,
#     rsl_rl_ppo_cfg.G1FlatPPOModNormRunnerCfg,
# )
_register(
    "Mjlab-LatentTracker-Flat-G1-ProjGravAnchorObs-TransformerVAE",
    flat_env_cfg.G1FlatProjGravAnchorObsTransformerVAEEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-ProjGravAnchorEEObs-TransformerVAE",
    flat_env_cfg.G1FlatProjGravAnchorEEObsTransformerVAEEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-ProjGravAnchorEEObs-TransformerVAE-SplitBodyReward",
    flat_env_cfg.G1FlatProjGravAnchorEEObsTransformerVAESplitBodyRewardEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
# _register(
#     "Mjlab-LatentTracker-Flat-G1-ProjGravAnchorEEObs-DirectRefMotion-NMMLP",
#     flat_env_cfg.G1FlatProjGravAnchorEEObsDirectRefMotionEnvCfg,
#     rsl_rl_ppo_cfg.G1FlatPPOModNormRunnerCfg,
# )
_register(
    "Mjlab-LatentTracker-Flat-G1-ProjGravAnchorEEObs-DirectRefMotion",
    flat_env_cfg.G1FlatProjGravAnchorEEObsDirectRefMotionEnvCfg,
    rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
)
_register(
    "Mjlab-LatentTracker-Flat-G1-Low-Freq",
    flat_env_cfg.G1FlatLowFreqEnvCfg,
    rsl_rl_ppo_cfg.G1FlatLowFreqPPORunnerCfg,
)
