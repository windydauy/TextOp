#!/usr/bin/env python3
"""Export a MotionAE encoder checkpoint to ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


DEFAULT_PROJECT_ROOT = Path("/home/humanoid/yzh/TextOp/motion_ae")
DEFAULT_RUN_DIR = Path(
    "/home/humanoid/yzh/TextOp/motion_ae/outputs/motion_ae/"
    "2026-05-08_17-48-33_opti_clean_our_20"
)
DEFAULT_CONFIG_PATH = DEFAULT_RUN_DIR / "params" / "config.yaml"
DEFAULT_CKPT_PATH = DEFAULT_RUN_DIR / "checkpoints" / "checkpoint_epoch29999.pt"
DEFAULT_STATS_PATH = DEFAULT_RUN_DIR / "artifacts" / "stats.npz"
DEFAULT_OUTPUT_PATH = DEFAULT_RUN_DIR / "artifacts" / "motion_ae_encoder_z_c.onnx"


class MotionAEEncoderOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, latent_mode: str) -> None:
        super().__init__()
        self.model = model
        self.latent_mode = latent_mode

    def forward(self, motion_window: torch.Tensor) -> torch.Tensor:
        z_dequant, z_d, info = self.model.encode(motion_window)
        if self.latent_mode == "z_dequant":
            return z_dequant
        if self.latent_mode == "z_d":
            return z_d.float()
        return info["z_c"]


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not find a MotionAE model state_dict in checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--stats_path", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--latent_mode", choices=("z_c", "z_dequant", "z_d"), default="z_c")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from motion_ae.config import load_config
    from motion_ae.models.autoencoder import MotionAutoEncoder
    from motion_ae.utils.normalization import FeatureNormalizer

    config_path = args.config_path.expanduser().resolve()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    stats_path = args.stats_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    for label, path in {
        "project root": project_root,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "stats": stats_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"MotionAE {label} not found: {path}")

    cfg = load_config(str(config_path))
    normalizer = FeatureNormalizer.load(str(stats_path), eps=cfg.normalization.eps)
    feature_dim = int(normalizer.mean.shape[0])
    device = torch.device(args.device)

    model = MotionAutoEncoder(
        feature_dim=feature_dim,
        window_size=int(cfg.window_size),
        encoder_hidden_dims=cfg.model.encoder_hidden_dims,
        decoder_hidden_dims=cfg.model.decoder_hidden_dims,
        ifsq_levels=cfg.model.ifsq_levels,
        activation=cfg.model.activation,
        use_layer_norm=cfg.model.use_layer_norm,
    )
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(device).eval()

    wrapper = MotionAEEncoderOnnxWrapper(model, args.latent_mode).to(device).eval()
    dummy_input = torch.from_numpy(
        np.zeros((1, int(cfg.window_size), feature_dim), dtype=np.float32)
    ).to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=["motion_window"],
            output_names=[args.latent_mode],
            dynamic_axes={
                "motion_window": {0: "batch"},
                args.latent_mode: {0: "batch"},
            },
            opset_version=args.opset,
        )

    print(f"Exported MotionAE encoder ONNX to: {output_path}")


if __name__ == "__main__":
    main()
