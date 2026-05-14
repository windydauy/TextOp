from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_SRC = REPO_ROOT / "TextOpTracker" / "source" / "textop_tracker"
if str(TRACKER_SRC) not in sys.path:
    sys.path.insert(0, str(TRACKER_SRC))


def test_motion_transformer_vae_adapter_encodes_real_motion_to_z_c() -> None:
    adapter_path = (
        TRACKER_SRC
        / "textop_tracker"
        / "tasks"
        / "tracking"
        / "mdp"
        / "motion_transformer_vae_latent.py"
    )
    spec = importlib.util.spec_from_file_location("motion_transformer_vae_latent", adapter_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MotionTransformerVAELatentAdapter = module.MotionTransformerVAELatentAdapter

    run_dir = (
        REPO_ROOT
        / "motion_ae"
        / "outputs"
        / "transformer_vae"
        / "2026-05-11_00-19-00_optitrack_npz_trip_filtered"
    )
    motion_path = (
        REPO_ROOT
        / "trip_npz_filtered"
        / "converted_0110"
        / "BALANCE_001_Skeleton 006_z_up_x_forward_gym_1"
        / "motion.npz"
    )

    adapter = MotionTransformerVAELatentAdapter(
        project_root=str(REPO_ROOT / "motion_ae" / "transformer_vae"),
        config_path=str(run_dir / "params" / "config.yaml"),
        checkpoint_path=str(run_dir / "checkpoints" / "best_model.pt"),
        stats_path=str(run_dir / "artifacts" / "stats.npz"),
        device="cpu",
        latent_mode="z_c",
        batch_size=64,
    )

    latents = adapter.encode_motion_file(str(motion_path))

    assert adapter.window_size == 10
    assert adapter.feature_dim == 70
    assert adapter.flat_latent_dim == 128
    assert latents.shape == (533, 128)
    assert latents.dtype == torch.float32
    assert torch.isfinite(latents).all()
