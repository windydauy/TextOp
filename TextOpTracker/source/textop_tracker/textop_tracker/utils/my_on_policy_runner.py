import os
import pickle
from typing import Optional

import numpy as np
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from textop_tracker.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _cap_probabilities_over_uniform(probabilities, max_prob_over_uniform):
    if max_prob_over_uniform is None or max_prob_over_uniform <= 0:
        return probabilities
    probabilities = np.asarray(probabilities, dtype=np.float64)
    num_items = probabilities.size
    max_prob = float(max_prob_over_uniform) / float(num_items)
    uniform_prob = 1.0 / float(num_items)
    if max_prob < uniform_prob:
        raise ValueError(
            "max_prob_over_uniform must be >= 1.0 because probabilities must sum to 1, "
            f"got {max_prob_over_uniform}"
        )
    if max_prob >= 1.0:
        return probabilities.astype(np.float32)

    order = np.argsort(-probabilities)
    sorted_probs = probabilities[order]
    prefix_before = np.concatenate([[0.0], np.cumsum(sorted_probs)[:-1]])
    capped_counts = np.arange(num_items, dtype=np.float64)
    remaining_mass = 1.0 - capped_counts * max_prob
    remaining_base = np.maximum(1.0 - prefix_before, 1e-12)
    scales = remaining_mass / remaining_base
    valid = (remaining_mass >= 0.0) & (sorted_probs * scales <= max_prob + 1e-12)
    if np.any(valid):
        capped_count = int(np.flatnonzero(valid)[0])
        capped_sorted = sorted_probs.copy()
        if capped_count > 0:
            capped_sorted[:capped_count] = max_prob
        capped_sorted[capped_count:] = sorted_probs[capped_count:] * scales[capped_count]
    else:
        capped_sorted = np.full_like(sorted_probs, uniform_prob)

    capped = np.empty_like(capped_sorted)
    capped[order] = capped_sorted
    capped = capped / max(capped.sum(), 1e-12)
    return capped.astype(np.float32)


# class MyOnPolicyRunner(OnPolicyRunner):
#     def save(self, path: str, infos=None):
#         """Save the model and training information."""
#         super().save(path, infos)
#         if self.logger_type in ["wandb"]:
#             policy_path = path.split("model")[0]
#             filename = policy_path.split("/")[-2] + ".onnx"
#             export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
#             attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
#             wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: Optional[str] = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        unwrapped_env = self.env.unwrapped

        policy_path = path.split("model")[0]
        onnx_filename = "latest.onnx"
        export_motion_policy_as_onnx(
            unwrapped_env, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=onnx_filename
        )
        if self.logger_type in ["wandb"]:
            attach_onnx_metadata(unwrapped_env, wandb.run.name, path=policy_path, filename=onnx_filename)
            wandb.save(policy_path + onnx_filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

        # For DEBUG:
        if unwrapped_env.command_manager.get_term("motion").cfg.enable_adaptive_sampling:
            motion_command = unwrapped_env.command_manager.get_term("motion")
            if motion_command.cfg.ads_type == "gear_sonic":
                adpsam_count = {
                    "ads_type": "gear_sonic",
                    "gear_sonic_bins": motion_command.gear_sonic_bins.cpu().numpy(),
                    "gear_sonic_bin_weights": motion_command.gear_sonic_bin_weights.cpu().numpy(),
                    "gear_sonic_num_episodes": motion_command.gear_sonic_num_episodes.cpu().numpy(),
                    "gear_sonic_num_failures": motion_command.gear_sonic_num_failures.cpu().numpy(),
                    "gear_sonic_failure_rate": motion_command.gear_sonic_failure_rate.cpu().numpy(),
                    "gear_sonic_sampling_probabilities": (
                        motion_command.gear_sonic_sampling_probabilities.cpu().numpy()
                    ),
                    "gear_sonic_bin_size": motion_command.cfg.gear_sonic_bin_size,
                    "gear_sonic_uniform_sampling_rate": (
                        motion_command.cfg.gear_sonic_uniform_sampling_rate
                    ),
                    "gear_sonic_failure_rate_max_over_mean": (
                        motion_command.cfg.gear_sonic_failure_rate_max_over_mean
                    ),
                    "gear_sonic_pre_failure_sample_window": (
                        motion_command.cfg.gear_sonic_pre_failure_sample_window
                    ),
                }
                pickle.dump(adpsam_count, open(path[:-len(".pt")] + "-adpsam_count.pkl", "wb"))
                return

            fail_count = motion_command.failed_motion_count.cpu().numpy()
            success_count = motion_command.success_motion_count.cpu().numpy()
            total_count = fail_count + success_count
            p_fail = fail_count / (total_count + 1e-8)
            p_fail_sample_v2 = (p_fail**motion_command.cfg.adaptive_beta)
            p_fail_sample_v2 = p_fail_sample_v2 / (p_fail_sample_v2.sum() + 1e-8)
            sampling_probabilities_v2 = (
                p_fail_sample_v2 * (1 - motion_command.cfg.adaptive_uniform_ratio) +
                motion_command.cfg.adaptive_uniform_ratio /
                float(motion_command.num_motion)
            )
            max_prob_over_uniform = motion_command.cfg.max_prob_over_uniform
            sampling_probabilities_v2_capped = _cap_probabilities_over_uniform(
                sampling_probabilities_v2, max_prob_over_uniform)
            adpsam_count = {
                "failed_motion_count": fail_count,
                "success_motion_count": success_count,
                "p_fail": p_fail,
                "p_fail_sample_v2": p_fail_sample_v2,
                "sampling_probabilities_v2": sampling_probabilities_v2,
                "max_prob_over_uniform": max_prob_over_uniform,
                "sampling_probabilities_v2_capped": sampling_probabilities_v2_capped,
            }
            # (adpsam_count, step=self.current_learning_iteration)
            pickle.dump(adpsam_count, open(path[:-len(".pt")] + "-adpsam_count.pkl", "wb"))
