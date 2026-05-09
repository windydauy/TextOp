from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


class MotionAELatentAdapter:
    """Encode tracker reference motions into MotionAE latents."""

    def __init__(
        self,
        *,
        project_root: str,
        config_path: str,
        checkpoint_path: str,
        stats_path: str,
        device: str | torch.device,
        latent_mode: str = "z_dequant",
        batch_size: int = 4096,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.stats_path = Path(stats_path).expanduser().resolve()
        self.device = torch.device(device)
        self.latent_mode = latent_mode
        self.batch_size = int(batch_size)

        self._validate_paths()
        if self.batch_size <= 0:
            raise ValueError(f"motion_ae batch_size must be positive, got {batch_size}")
        if self.latent_mode not in {"z_dequant", "z_c", "z_d"}:
            raise ValueError(f"Unsupported MotionAE latent mode: {latent_mode}")

        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        from motion_ae.config import load_config
        from motion_ae.models.autoencoder import MotionAutoEncoder
        from motion_ae.utils.normalization import FeatureNormalizer

        self.cfg = load_config(str(self.config_path))
        self.normalizer = FeatureNormalizer.load(
            str(self.stats_path),
            eps=self.cfg.normalization.eps,
        )
        self.window_size = int(self.cfg.window_size)
        self.feature_dim = int(self.normalizer.mean.shape[0])
        self.latent_dim = int(self.cfg.model.latent_dim)

        self.model = MotionAutoEncoder(
            feature_dim=self.feature_dim,
            window_size=self.window_size,
            encoder_hidden_dims=self.cfg.model.encoder_hidden_dims,
            decoder_hidden_dims=self.cfg.model.decoder_hidden_dims,
            ifsq_levels=self.cfg.model.ifsq_levels,
            activation=self.cfg.model.activation,
            use_layer_norm=self.cfg.model.use_layer_norm,
        )
        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._mean = torch.as_tensor(self.normalizer.mean, dtype=torch.float32, device=self.device)
        self._std = torch.as_tensor(self.normalizer.std, dtype=torch.float32, device=self.device)

    def _validate_paths(self) -> None:
        for label, path in {
            "MotionAE project root": self.project_root,
            "MotionAE config": self.config_path,
            "MotionAE checkpoint": self.checkpoint_path,
            "MotionAE stats": self.stats_path,
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

    def encode_motion_file(self, motion_file: str) -> torch.Tensor:
        """Load one `motion.npz` and return per-frame latent commands."""
        with np.load(motion_file) as npz_data:
            arrays = {key: npz_data[key] for key in npz_data.files}
        return self.encode_npz_data(arrays)

    def encode_npz_data(self, npz_data: Mapping[str, Any]) -> torch.Tensor:
        """Return one latent per frame using clamped future windows."""
        from motion_ae.feature_builder import build_features

        features, slices = build_features(
            npz_data,
            self.cfg.npz_keys,
            self.cfg.pelvis,
            debug=False,
        )
        if slices.total_dim != self.feature_dim:
            raise ValueError(
                f"MotionAE feature dim mismatch: checkpoint expects {self.feature_dim}, "
                f"motion produced {slices.total_dim}"
            )
        if features.shape[0] == 0:
            raise ValueError("Cannot encode empty motion")

        features_t = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        normalized = (features_t - self._mean) / self._std
        windows = self._future_windows(normalized)

        latents: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in windows.split(self.batch_size, dim=0):
                z_dequant, z_d, info = self.model.encode(batch)
                if self.latent_mode == "z_dequant":
                    latent = z_dequant
                elif self.latent_mode == "z_c":
                    latent = info["z_c"]
                else:
                    latent = z_d
                latents.append(latent.float())

        return torch.cat(latents, dim=0)

    def _future_windows(self, features: torch.Tensor) -> torch.Tensor:
        """Build `[T, window_size, feature_dim]` windows from `t:t+window_size`."""
        frame_count = int(features.shape[0])
        frame_ids = torch.arange(frame_count, device=self.device)
        offsets = torch.arange(self.window_size, device=self.device)
        indices = torch.clamp(frame_ids[:, None] + offsets[None, :], max=frame_count - 1)
        flat = features.index_select(0, indices.reshape(-1))
        return flat.reshape(frame_count, self.window_size, self.feature_dim)
