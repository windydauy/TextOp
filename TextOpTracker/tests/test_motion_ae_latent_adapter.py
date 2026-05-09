from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_SRC = REPO_ROOT / "TextOpTracker" / "source" / "textop_tracker"
if str(TRACKER_SRC) not in sys.path:
    sys.path.insert(0, str(TRACKER_SRC))


def test_motion_ae_adapter_encodes_real_motion_to_z_dequant() -> None:
    adapter_path = (
        TRACKER_SRC
        / "textop_tracker"
        / "tasks"
        / "tracking"
        / "mdp"
        / "motion_ae_latent.py"
    )
    spec = importlib.util.spec_from_file_location("motion_ae_latent", adapter_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MotionAELatentAdapter = module.MotionAELatentAdapter

    run_dir = REPO_ROOT / "motion_ae" / "outputs" / "motion_ae" / "2026-05-08_17-48-33_opti_clean_our_20"
    motion_path = next((REPO_ROOT / "motion_ae" / "optitrack_npz_filtered_all").glob("*/motion.npz"))

    adapter = MotionAELatentAdapter(
        project_root=str(REPO_ROOT / "motion_ae"),
        config_path=str(run_dir / "params" / "config.yaml"),
        checkpoint_path=str(run_dir / "checkpoints" / "checkpoint_epoch29999.pt"),
        stats_path=str(run_dir / "artifacts" / "stats.npz"),
        device="cpu",
        latent_mode="z_dequant",
        batch_size=64,
    )

    latents = adapter.encode_motion_file(str(motion_path))

    assert adapter.window_size == 10
    assert latents.shape == (1534, 32)
    assert latents.dtype == torch.float32
    assert torch.isfinite(latents).all()
    assert latents.min() >= -1.0
    assert latents.max() <= 1.0
