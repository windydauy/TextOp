from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_motion_transformer_vae_deploy_task_and_scripts_are_wired() -> None:
    deploy_src = (REPO_ROOT / "TextOpTracker" / "scripts" / "deploy_mujoco.py").read_text(encoding="utf-8")
    deploy_script_src = (REPO_ROOT / "scripts" / "deploy_motion_transformer_vae_zc_mujoco.sh").read_text(
        encoding="utf-8"
    )
    export_script_src = (REPO_ROOT / "scripts" / "export_motion_transformer_vae_onnx.py").read_text(
        encoding="utf-8"
    )

    assert "Tracking-Flat-G1-ProjGravAnchorObs-TransformerVAE-NMMLP-v0" in deploy_src
    assert "MotionTransformerVAEOnnxLatentAdapter" in deploy_src
    assert "motion_transformer_vae_encoder_onnx_path" in deploy_src
    assert "MOTION_TRANSFORMER_VAE_LATENT_DIM" in deploy_src
    assert "export_motion_transformer_vae_onnx.py" in deploy_script_src
    assert "motion_transformer_vae_zc_npz_filtered/latest.onnx" in deploy_script_src
    assert "best_model.pt" in deploy_script_src
    assert "torch.onnx.export" in export_script_src
    assert "z_c" in export_script_src
