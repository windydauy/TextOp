from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_split_body_transformer_vae_eeobs_reward_task_is_wired() -> None:
    tracking_cfg_src = (
        REPO_ROOT
        / "TextOpTracker"
        / "source"
        / "textop_tracker"
        / "textop_tracker"
        / "tasks"
        / "tracking"
        / "tracking_env_cfg.py"
    ).read_text(encoding="utf-8")
    rewards_src = (
        REPO_ROOT
        / "TextOpTracker"
        / "source"
        / "textop_tracker"
        / "textop_tracker"
        / "tasks"
        / "tracking"
        / "mdp"
        / "rewards.py"
    ).read_text(encoding="utf-8")
    flat_cfg_src = (
        REPO_ROOT
        / "TextOpTracker"
        / "source"
        / "textop_tracker"
        / "textop_tracker"
        / "tasks"
        / "tracking"
        / "config"
        / "g1"
        / "flat_env_cfg.py"
    ).read_text(encoding="utf-8")
    registry_src = (
        REPO_ROOT
        / "TextOpTracker"
        / "source"
        / "textop_tracker"
        / "textop_tracker"
        / "tasks"
        / "tracking"
        / "config"
        / "g1"
        / "__init__.py"
    ).read_text(encoding="utf-8")

    task_id = "Tracking-Flat-G1-ProjGravAnchorEEObs-TransformerVAE-SplitBodyReward-NMMLP-v0"
    assert "G1FlatProjGravAnchorEEObsTransformerVAESplitBodyRewardEnvCfg" in flat_cfg_src
    assert "ProjGravAnchorEEObsSplitBodyTransformerVAERewardsCfg" in flat_cfg_src
    assert task_id in registry_src

    assert "_ALL_TRACKED_BODY_COUNT = len(_NON_EE_BODY_NAMES) + len(_EE_BODY_NAMES)" in tracking_cfg_src
    assert "_NON_EE_BODY_REWARD_WEIGHT = len(_NON_EE_BODY_NAMES) / _ALL_TRACKED_BODY_COUNT" in tracking_cfg_src
    assert "_EE_BODY_REWARD_WEIGHT = len(_EE_BODY_NAMES) / _ALL_TRACKED_BODY_COUNT" in tracking_cfg_src

    reward_cfg_body = tracking_cfg_src.split(
        "class ProjGravAnchorEEObsSplitBodyTransformerVAERewardsCfg", 1
    )[1].split("\n\n@configclass", 1)[0]
    for disabled_reward in ("feet_slide", "soft_landing", "overspeed", "overeffort"):
        assert f"{disabled_reward} = None" in reward_cfg_body

    for reward_name in (
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    ):
        assert reward_name in reward_cfg_body
    assert reward_cfg_body.count("weight=_NON_EE_BODY_REWARD_WEIGHT") == 4
    assert reward_cfg_body.count('"body_names": _NON_EE_BODY_NAMES') == 4

    for reward_name in (
        "motion_ee_body_pos",
        "motion_ee_body_ori",
        "motion_ee_body_lin_vel",
        "motion_ee_body_ang_vel",
        "motion_ee_global_body_pos",
        "motion_ee_global_body_ori",
    ):
        assert reward_name in reward_cfg_body
    assert reward_cfg_body.count("weight=_EE_BODY_REWARD_WEIGHT") == 4
    assert reward_cfg_body.count('"body_names": _EE_BODY_NAMES') == 6
    assert "motion_ee_body_pos = RewTerm(\n        func=mdp.motion_future_obs_relative_body_position_error_exp,\n        weight=_EE_BODY_REWARD_WEIGHT" in reward_cfg_body
    assert "motion_ee_body_ori = RewTerm(\n        func=mdp.motion_future_obs_relative_body_orientation_error_exp,\n        weight=_EE_BODY_REWARD_WEIGHT" in reward_cfg_body
    assert "motion_ee_global_body_pos = RewTerm(\n        func=mdp.motion_future_obs_global_body_position_error_exp,\n        weight=1.0" in reward_cfg_body
    assert "motion_ee_global_body_ori = RewTerm(\n        func=mdp.motion_future_obs_global_body_orientation_error_exp,\n        weight=1.0" in reward_cfg_body

    for future_obs_reward in (
        "motion_future_obs_relative_body_position_error_exp",
        "motion_future_obs_relative_body_orientation_error_exp",
        "motion_future_obs_global_body_position_error_exp",
        "motion_future_obs_global_body_orientation_error_exp",
        "motion_future_obs_global_body_linear_velocity_error_exp",
        "motion_future_obs_global_body_angular_velocity_error_exp",
    ):
        assert future_obs_reward in rewards_src
        assert future_obs_reward in reward_cfg_body

    assert "command.future_body_pos_quat_w(body_indexes)" in rewards_src
    assert "body_indexes = [command.cfg.body_names.index(name) for name in body_names]" in rewards_src
    assert "command.motion_anchor_pos.view(num_envs, future_steps, 3)" in rewards_src
    assert "command.motion_anchor_quat.view(num_envs, future_steps, 4)" in rewards_src

    relative_ref_body = rewards_src.split("def _future_obs_relative_body_reference_w", 1)[1].split(
        "\n\ndef reward_cond_on_pfail", 1
    )[0]
    assert "motion_ee_pos_b" in relative_ref_body
    assert "motion_ee_ori_b" in relative_ref_body
    assert "motion_anchor_pos_b" in relative_ref_body
    assert "motion_anchor_ori_b" in relative_ref_body
    assert "robot_anchor_pos_w_obs" in relative_ref_body
    assert "robot_anchor_ori_w" in relative_ref_body
    assert "obs_motion_body_pos_b_future" in relative_ref_body
    assert "obs_motion_body_ori_b_future" in relative_ref_body
    assert "obs_motion_anchor_pos_b_future" in relative_ref_body
    assert "obs_motion_anchor_ori_b_future" in relative_ref_body
    assert "obs_robot_anchor_pos_w" in relative_ref_body
    assert "obs_robot_anchor_ori_w" in relative_ref_body
    assert "command.future_body_pos_quat_w" not in relative_ref_body
    assert "body_pos_relative_w" in relative_ref_body
    assert "body_quat_relative_w" in relative_ref_body
    assert "delta_pos_w[..., 2] = anchor_pos_w[:, None, 2]" in relative_ref_body
    assert "delta_yaw_quat = yaw_quat" in relative_ref_body

    relative_pos_body = rewards_src.split("def motion_future_obs_relative_body_position_error_exp", 1)[1].split(
        "\n\ndef motion_future_obs_relative_body_orientation_error_exp", 1
    )[0]
    relative_ori_body = rewards_src.split("def motion_future_obs_relative_body_orientation_error_exp", 1)[1].split(
        "\n\ndef motion_future_obs_global_body_position_error_exp", 1
    )[0]
    assert "_future_obs_relative_body_reference_w" in relative_pos_body
    assert "_future_ee_reference_from_obs_layout" not in relative_pos_body
    assert "ref[\"body_pos_relative_w\"] - command.robot_body_pos_w[:, body_indexes]" in relative_pos_body
    assert "_future_obs_relative_body_reference_w" in relative_ori_body
    assert "_future_ee_reference_from_obs_layout" not in relative_ori_body
    assert "quat_error_magnitude(" in relative_ori_body
    assert "ref[\"body_quat_relative_w\"], command.robot_body_quat_w[:, body_indexes]" in relative_ori_body
