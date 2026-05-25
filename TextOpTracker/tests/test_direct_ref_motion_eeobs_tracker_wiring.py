from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_direct_ref_motion_eeobs_task_and_script_are_wired() -> None:
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
    script_src = (REPO_ROOT / "scripts" / "train_direct_ref_motion_eeobs_g1_before_2023.sh").read_text(
        encoding="utf-8"
    )

    task_id = "Tracking-Flat-G1-ProjGravAnchorEEObs-DirectRefMotion-NMMLP-v0"
    assert "G1FlatProjGravAnchorEEObsDirectRefMotionEnvCfg" in flat_cfg_src
    assert "G1FlatProjGravAnchorEEObsTransformerVAEEnvCfg" in flat_cfg_src
    assert "self.commands.motion.motion_transformer_vae_enabled = False" in flat_cfg_src
    assert task_id in registry_src
    assert task_id in deploy_src
    assert f"--task={task_id}" in script_src
    assert 'MOTION_ROOT="${MOTION_ROOT:-/home/humanoid/yzh/TextOp/g1_before_2023}"' in script_src
    assert 'NUM_ENVS="${NUM_ENVS:-8192}"' in script_src
    assert "env.commands.motion.motion_ae_enabled=False" in script_src
    assert "env.commands.motion.motion_transformer_vae_enabled=False" in script_src
