from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_transformer_vae_eeobs_tracker_task_rewards_and_deploy_are_wired() -> None:
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
    deploy_src = (REPO_ROOT / "TextOpTracker" / "scripts" / "deploy_mujoco.py").read_text(encoding="utf-8")
    eval_src = (REPO_ROOT / "TextOpTracker" / "scripts" / "evaluate_mujoco_onnx.py").read_text(encoding="utf-8")

    task_id = "Tracking-Flat-G1-ProjGravAnchorEEObs-TransformerVAE-NMMLP-v0"
    assert "G1FlatProjGravAnchorEEObsTransformerVAEEnvCfg" in flat_cfg_src
    assert "ProjGravAnchorEEObsTransformerVAERewardsCfg" in tracking_cfg_src
    assert 'self.commands.motion.motion_transformer_vae_latent_mode = "z_c"' in flat_cfg_src
    assert task_id in registry_src
    assert task_id in deploy_src

    for reward_name in (
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "motion_ee_body_pos",
        "motion_ee_body_ori",
        "motion_ee_body_lin_vel",
        "motion_ee_body_ang_vel",
        "motion_ee_global_body_pos",
        "motion_ee_global_body_ori",
    ):
        assert reward_name in tracking_cfg_src

    assert "motion_global_body_position_error_exp" in rewards_src
    assert "motion_global_body_orientation_error_exp" in rewards_src
    assert "_NON_EE_BODY_NAMES" in tracking_cfg_src
    assert tracking_cfg_src.count('"body_names": _NON_EE_BODY_NAMES') >= 4
    reward_cfg_body = tracking_cfg_src.split("class ProjGravAnchorEEObsTransformerVAERewardsCfg", 1)[1].split(
        "\n\n@configclass", 1
    )[0]
    for disabled_reward in ("feet_slide", "soft_landing", "overspeed", "overeffort"):
        assert f"{disabled_reward} = None" in reward_cfg_body
    assert "TASK_PROJ_GRAV_ANCHOR_EE_OBS_TRANSFORMER_VAE" in deploy_src
    assert "TASK_PROJ_GRAV_ANCHOR_EE_OBS_TRANSFORMER_VAE" in eval_src

    ee_terms_block = deploy_src.split("PROJ_GRAV_ANCHOR_EE_OBS_TERMS = (", 1)[1].split(")", 1)[0]
    assert ee_terms_block.index('"actions"') < ee_terms_block.index('"motion_ee_pos_b"')
    assert ee_terms_block.index('"motion_ee_pos_b"') < ee_terms_block.index('"motion_ee_ori_b"')
