from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

from latent_tracker.robots.g1 import G1_ACTION_SCALE, get_g1_robot_cfg
from latent_tracker.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from latent_tracker.tasks.tracking.mdp import MotionCommandCfg
from latent_tracker.tasks.tracking.tracking_env_cfg import (
    TRACKED_BODY_NAMES,
    make_tracking_env_cfg,
)


def _apply_g1(cfg: ManagerBasedRlEnvCfg, *, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = G1_ACTION_SCALE

    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    command.anchor_body_name = "torso_link"
    command.body_names = list(TRACKED_BODY_NAMES)

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = "torso_link"
    cfg.terminations["ee_body_pos"].params["body_names"] = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    cfg.viewer.body_name = "torso_link"

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        command.pose_range = {}
        command.velocity_range = {}
        command.sampling_mode = "start"
    return cfg


def G1FlatEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _apply_g1(make_tracking_env_cfg(), play=play)


def G1FlatWoStateEstimationEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = G1FlatEnvCfg(play=play)
    actor_terms = {
        key: value
        for key, value in cfg.observations["actor"].terms.items()
        if key not in {"motion_anchor_pos_b", "base_lin_vel"}
    }
    cfg.observations["actor"].terms = actor_terms
    return cfg


def G1FlatProjGravObsEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _apply_g1(make_tracking_env_cfg(observation_mode="proj_grav"), play=play)


def G1FlatProjGravObsEnvCfg_LargeHand(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return G1FlatProjGravObsEnvCfg(play=play)


def G1FlatProjGravObsEnvCfg_LargeHandHeavy(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return G1FlatProjGravObsEnvCfg(play=play)


def G1FlatPropPropObsEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _apply_g1(make_tracking_env_cfg(observation_mode="prop_prop"), play=play)


def G1FlatPrivPrivObsEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _apply_g1(make_tracking_env_cfg(observation_mode="priv_priv"), play=play)


def G1FlatNoisePrivObsEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _apply_g1(make_tracking_env_cfg(observation_mode="noise_priv"), play=play)


def G1FlatProjGravAnchorObsEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    return _apply_g1(make_tracking_env_cfg(observation_mode="proj_grav_anchor"), play=play)


def G1FlatProjGravAnchorObsMotionAEEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = G1FlatProjGravAnchorObsEnvCfg(play=play)
    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    command.future_steps = 10
    command.motion_ae_enabled = True
    command.motion_ae_latent_mode = "z_dequant"
    return cfg


def G1FlatProjGravAnchorEEObsEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _apply_g1(make_tracking_env_cfg(observation_mode="proj_grav_anchor_ee"), play=play)
    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    command.future_steps = 10
    command.motion_ae_enabled = True
    command.motion_ae_latent_mode = "z_c"
    return cfg


def G1FlatProjGravAnchorObsTransformerVAEEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = G1FlatProjGravAnchorObsEnvCfg(play=play)
    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    command.future_steps = 10
    command.motion_transformer_vae_enabled = True
    command.motion_transformer_vae_latent_mode = "z_c"
    return cfg


def G1FlatProjGravAnchorEEObsTransformerVAEEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = _apply_g1(
        make_tracking_env_cfg(
            observation_mode="proj_grav_anchor_ee",
            reward_mode="transformer_vae_ee",
        ),
        play=play,
    )
    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    command.future_steps = 10
    command.motion_transformer_vae_enabled = True
    command.motion_transformer_vae_latent_mode = "z_c"
    return cfg


def G1FlatProjGravAnchorEEObsTransformerVAESplitBodyRewardEnvCfg(
    *, play: bool = False
) -> ManagerBasedRlEnvCfg:
    cfg = _apply_g1(
        make_tracking_env_cfg(
            observation_mode="proj_grav_anchor_ee",
            reward_mode="split_body_transformer_vae_ee",
        ),
        play=play,
    )
    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    command.future_steps = 10
    command.motion_transformer_vae_enabled = True
    command.motion_transformer_vae_latent_mode = "z_c"
    return cfg


def G1FlatProjGravAnchorEEObsDirectRefMotionEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = G1FlatProjGravAnchorEEObsTransformerVAEEnvCfg(play=play)
    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    command.motion_transformer_vae_enabled = False
    return cfg


def G1FlatLowFreqEnvCfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = G1FlatEnvCfg(play=play)
    cfg.decimation = round(cfg.decimation / LOW_FREQ_SCALE)
    cfg.rewards["action_rate_l2"].weight *= LOW_FREQ_SCALE
    return cfg
