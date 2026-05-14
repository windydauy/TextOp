from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_motion_ae_zc_eeobs_tracker_task_and_script_are_wired() -> None:
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
    script_src = (REPO_ROOT / "scripts" / "train_motion_ae_zc_eeobs.sh").read_text(
        encoding="utf-8"
    )

    assert "ProjGravAnchorEEObsObservationsCfg" in tracking_cfg_src
    assert "motion_ee_pos_b" in tracking_cfg_src
    assert "motion_ee_ori_b" in tracking_cfg_src
    assert "G1FlatProjGravAnchorEEObsEnvCfg" in flat_cfg_src
    assert 'self.commands.motion.motion_ae_latent_mode = "z_c"' in flat_cfg_src
    assert "Tracking-Flat-G1-ProjGravAnchorEEObs-NMMLP-v0" in registry_src
    assert "--task=Tracking-Flat-G1-ProjGravAnchorEEObs-NMMLP-v0" in script_src
    assert "--num_envs=\"${NUM_ENVS}\"" in script_src
    assert "optitrack_npz_filtered_all" in script_src
