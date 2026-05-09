from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_motion_ae_tracker_task_and_script_are_wired() -> None:
    commands_src = (
        REPO_ROOT
        / "TextOpTracker"
        / "source"
        / "textop_tracker"
        / "textop_tracker"
        / "tasks"
        / "tracking"
        / "mdp"
        / "commands_multi.py"
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
    script_src = (REPO_ROOT / "scripts" / "train_motion_ae_zdequant.sh").read_text(encoding="utf-8")

    assert "MotionAELatentAdapter" in commands_src
    assert "motion_ae_enabled" in commands_src
    assert "motion_ae_latents_list" in commands_src
    assert "G1FlatProjGravAnchorObsMotionAEEnvCfg" in flat_cfg_src
    assert "Tracking-Flat-G1-ProjGravAnchorObs-MotionAE-NMMLP-v0" in registry_src
    assert "--num_envs=\"${NUM_ENVS:-8192}\"" in script_src
    assert "motion_ae_zdequant" in script_src
