from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


class MotionTransformerVAELatentAdapter:
    """Encode tracker reference motions into deterministic Transformer VAE latents."""

    def __init__(
        self,
        *,
        project_root: str,
        config_path: str,
        checkpoint_path: str,
        stats_path: str,
        device: str | torch.device,
        latent_mode: str = "z_c",
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
            raise ValueError(f"motion_transformer_vae batch_size must be positive, got {batch_size}")
        if self.latent_mode not in {"z_c", "mu"}:
            raise ValueError(f"Unsupported MotionTransformerVAE latent mode: {latent_mode}")

        for import_root in (self.project_root.parent, self.project_root):
            import_root_str = str(import_root)
            if import_root_str not in sys.path:
                sys.path.insert(0, import_root_str)

        from motion_ae.utils.normalization import FeatureNormalizer
        from transformer_vae.config import load_config
        from transformer_vae.scripts.common import build_model

        self.cfg = load_config(str(self.config_path))
        self.normalizer = FeatureNormalizer.load(
            str(self.stats_path),
            eps=self.cfg.normalization.eps,
        )
        self.window_size = int(self.cfg.window_size)
        self.feature_dim = int(self.normalizer.mean.shape[0])
        self.latent_size = int(self.cfg.model.latent_dim[0])
        self.latent_dim = int(self.cfg.model.latent_dim[-1])
        self.flat_latent_dim = self.latent_size * self.latent_dim

        self.model = build_model(self.cfg, self.feature_dim)
        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )
        state_dict = self._extract_state_dict(checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._mean = torch.as_tensor(self.normalizer.mean, dtype=torch.float32, device=self.device)
        self._std = torch.as_tensor(self.normalizer.std, dtype=torch.float32, device=self.device)

    def _validate_paths(self) -> None:
        for label, path in {
            "MotionTransformerVAE project root": self.project_root,
            "MotionTransformerVAE config": self.config_path,
            "MotionTransformerVAE checkpoint": self.checkpoint_path,
            "MotionTransformerVAE stats": self.stats_path,
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
        if isinstance(checkpoint, Mapping):
            for key in ("model_state_dict", "vae", "state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, Mapping):
                    return value
            if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                return checkpoint
        raise ValueError("Could not find a Transformer VAE model state_dict in checkpoint")

    def encode_motion_file(self, motion_file: str) -> torch.Tensor:
        """Load one `motion.npz` and return one latent command per frame."""
        with np.load(motion_file) as npz_data:
            arrays = {key: npz_data[key] for key in npz_data.files}
        return self.encode_npz_data(arrays)

    def encode_npz_data(self, npz_data: Mapping[str, Any]) -> torch.Tensor:
        """Return deterministic `[T, latent_size * latent_dim]` latents from clamped future windows."""
        from motion_ae.feature_builder import build_features

        features, slices = build_features(
            npz_data,
            self.cfg.npz_keys,
            self.cfg.pelvis,
            debug=False,
        )
        if slices.total_dim != self.feature_dim:
            raise ValueError(
                "MotionTransformerVAE feature dim mismatch: checkpoint expects "
                f"{self.feature_dim}, motion produced {slices.total_dim}"
            )
        if features.shape[0] == 0:
            raise ValueError("Cannot encode empty motion")

        features_t = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        normalized = (features_t - self._mean) / self._std
        windows = self._future_windows(normalized)

        latents: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in windows.split(self.batch_size, dim=0):
                z_c, _dist, mu, _logvar = self.model.encode(batch, sample=False)
                latent = mu if self.latent_mode == "mu" else z_c
                latent = latent.permute(1, 0, 2).reshape(batch.shape[0], self.flat_latent_dim)
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
