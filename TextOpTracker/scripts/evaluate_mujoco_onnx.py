#!/usr/bin/env python3
"""Evaluate an exported policy ONNX in MuJoCo over a folder of motion.npz files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import mujoco
import numpy as np

import deploy_mujoco as dm


EVAL_BODY_NAMES = [
    "pelvis",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

SUCCESS_METRIC_KEYS = (
    "global_mpjpe",
    "local_mpjpe",
    "anchor_global_pos_error",
    "anchor_local_pos_error",
)

FAIL_REASON_KEYS = (
    "nan_or_inf",
    "pelvis_height",
    "local_mpjpe",
)

INTERNAL_METRIC_KEYS = (
    "_joint_torque_abs_sum",
    "_joint_torque_abs_sq_sum",
    "_joint_torque_count",
)


def resolve_motion_files(path_arg: str, limit: int | None = None) -> list[str]:
    path = Path(path_arg).expanduser()
    if path.is_file():
        files = [path]
    elif any(ch in str(path) for ch in "*?[]"):
        files = [Path(p) for p in sorted(path.parent.glob(path.name))]
    else:
        files = sorted(path.glob("**/motion.npz"))
    files = [p for p in files if p.is_file() and p.name == "motion.npz"]
    if limit is not None and limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No motion.npz files found from {path_arg}")
    return [str(p) for p in files]


def quat_error_rad(q_ref: np.ndarray, q_robot: np.ndarray) -> np.ndarray:
    q_ref_t = dm.torch.from_numpy(q_ref.astype(np.float32))
    q_robot_t = dm.torch.from_numpy(q_robot.astype(np.float32))
    dot = (q_ref_t * q_robot_t).sum(dim=-1).abs().clamp(max=1.0)
    return (2.0 * dm.torch.acos(dot)).cpu().numpy()


def local_positions(pos_w: np.ndarray, quat_w: np.ndarray, anchor_pos_w: np.ndarray, anchor_quat_w: np.ndarray) -> np.ndarray:
    anchor_pos = np.repeat(anchor_pos_w.reshape(1, 3), pos_w.shape[0], axis=0)
    anchor_quat = np.repeat(anchor_quat_w.reshape(1, 4), pos_w.shape[0], axis=0)
    pos_b, _ = dm.subtract_frame_transforms(
        dm.torch.from_numpy(anchor_pos.astype(np.float32)),
        dm.torch.from_numpy(anchor_quat.astype(np.float32)),
        dm.torch.from_numpy(pos_w.astype(np.float32)),
        dm.torch.from_numpy(quat_w.astype(np.float32)),
    )
    return pos_b.cpu().numpy()


class MetricAccumulator:
    def __init__(self) -> None:
        self.frames = 0
        self.global_mpjpe_sum = 0.0
        self.local_mpjpe_sum = 0.0
        self.anchor_global_pos_sum = 0.0
        self.anchor_local_pos_sum = 0.0
        self.dof_pos_error_sum = 0.0
        self.joint_torque_abs_sum = 0.0
        self.joint_torque_abs_sq_sum = 0.0
        self.joint_torque_count = 0
        self.eval_body_global_pos_sum = {name: 0.0 for name in EVAL_BODY_NAMES}
        self.eval_body_global_ori_sum = {name: 0.0 for name in EVAL_BODY_NAMES}

    def update(self, values: dict[str, float]) -> None:
        self.frames += 1
        self.global_mpjpe_sum += values["global_mpjpe"]
        self.local_mpjpe_sum += values["local_mpjpe"]
        self.anchor_global_pos_sum += values["anchor_global_pos_error"]
        self.anchor_local_pos_sum += values["anchor_local_pos_error"]
        self.dof_pos_error_sum += values["mean_dof_position_error"]
        self.joint_torque_abs_sum += values["_joint_torque_abs_sum"]
        self.joint_torque_abs_sq_sum += values["_joint_torque_abs_sq_sum"]
        self.joint_torque_count += int(values["_joint_torque_count"])
        for name in EVAL_BODY_NAMES:
            self.eval_body_global_pos_sum[name] += values[f"{name}_global_pos_error"]
            self.eval_body_global_ori_sum[name] += values[f"{name}_global_ori_error_rad"]

    def summary(self) -> dict[str, float | dict[str, float]]:
        if self.frames == 0:
            raise RuntimeError("No frames were evaluated")
        inv = 1.0 / self.frames
        torque_mean = 0.0
        torque_std = 0.0
        if self.joint_torque_count > 0:
            torque_mean = self.joint_torque_abs_sum / self.joint_torque_count
            torque_var = self.joint_torque_abs_sq_sum / self.joint_torque_count - torque_mean**2
            torque_std = float(np.sqrt(max(torque_var, 0.0)))
        result: dict[str, float | dict[str, float]] = {
            "frames": self.frames,
            "global_mpjpe": self.global_mpjpe_sum * inv,
            "local_mpjpe": self.local_mpjpe_sum * inv,
            "anchor_global_pos_error": self.anchor_global_pos_sum * inv,
            "anchor_local_pos_error": self.anchor_local_pos_sum * inv,
            "mean_dof_position_error": self.dof_pos_error_sum * inv,
            "mean_joint_torque": float(torque_mean),
            "joint_torque_std": torque_std,
            "_joint_torque_abs_sum": float(self.joint_torque_abs_sum),
            "_joint_torque_abs_sq_sum": float(self.joint_torque_abs_sq_sum),
            "_joint_torque_count": int(self.joint_torque_count),
            "eval_body_global_pos_error": {
                name: self.eval_body_global_pos_sum[name] * inv for name in EVAL_BODY_NAMES
            },
            "eval_body_global_ori_error_rad": {
                name: self.eval_body_global_ori_sum[name] * inv for name in EVAL_BODY_NAMES
            },
        }
        return result


class TerminationTracker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.counts = {key: 0 for key in FAIL_REASON_KEYS}

    def update(self, values: dict[str, float], data: mujoco.MjData) -> str | None:
        consecutive_fail_frames = max(int(self.args.consecutive_fail_frames), 1)
        reasons = {
            "nan_or_inf": self._has_nonfinite(values, data),
            "pelvis_height": values["pelvis_global_height"] < self.args.pelvis_height_min,
            "local_mpjpe": values["local_mpjpe"] > self.args.local_mpjpe_fail,
        }
        for reason, failed in reasons.items():
            self.counts[reason] = self.counts[reason] + 1 if failed else 0
            if self.counts[reason] >= consecutive_fail_frames:
                return reason
        return None

    @staticmethod
    def _has_nonfinite(values: dict[str, float], data: mujoco.MjData) -> bool:
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            return True
        return not all(np.isfinite(value) for value in values.values())


def make_latent_adapters(task: str, args: argparse.Namespace):
    motion_ae_adapter = None
    motion_transformer_vae_adapter = None
    if task in (dm.TASK_PROJ_GRAV_ANCHOR_OBS_MOTION_AE, dm.TASK_PROJ_GRAV_ANCHOR_EE_OBS):
        motion_ae_adapter = dm.MotionAEOnnxLatentAdapter(
            project_root=args.motion_ae_project_root,
            config_path=args.motion_ae_config_path,
            stats_path=args.motion_ae_stats_path,
            encoder_onnx_path=args.motion_ae_encoder_onnx_path,
            batch_size=args.motion_ae_batch_size,
        )
    elif task in (
        dm.TASK_PROJ_GRAV_ANCHOR_OBS_TRANSFORMER_VAE,
        dm.TASK_PROJ_GRAV_ANCHOR_EE_OBS_TRANSFORMER_VAE,
        dm.TASK_PROJ_GRAV_ANCHOR_EE_OBS_ONESTEP_TRANSFORMER_VAE,
    ):
        motion_transformer_vae_adapter = dm.MotionTransformerVAEOnnxLatentAdapter(
            project_root=args.motion_transformer_vae_project_root,
            config_path=args.motion_transformer_vae_config_path,
            stats_path=args.motion_transformer_vae_stats_path,
            encoder_onnx_path=args.motion_transformer_vae_encoder_onnx_path,
            batch_size=args.motion_transformer_vae_batch_size,
        )
    return motion_ae_adapter, motion_transformer_vae_adapter


def compute_frame_metrics(
    sim_data,
    motion_loader: dm.MotionLoader,
    t: int,
    eval_indexes: list[int],
    joint_torque: np.ndarray,
) -> dict[str, float]:
    step_idx = min(t, motion_loader.T - 1)
    ref_body_indexes = [motion_loader.motion_body_index(name) for name in motion_loader.body_names]
    robot_pos = np.stack([sim_data.body(name).xpos.copy() for name in motion_loader.body_names], axis=0)
    robot_quat = np.stack([sim_data.body(name).xquat.copy() for name in motion_loader.body_names], axis=0)
    ref_pos = motion_loader.body_pos[step_idx, ref_body_indexes]
    ref_quat = motion_loader.body_ori[step_idx, ref_body_indexes]

    global_errors = np.linalg.norm(robot_pos - ref_pos, axis=-1)
    robot_anchor_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy()
    robot_anchor_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy()
    ref_anchor_pos = motion_loader.body_pos[step_idx, motion_loader.anchor_body_index]
    ref_anchor_quat = motion_loader.body_ori[step_idx, motion_loader.anchor_body_index]

    robot_local_pos = local_positions(robot_pos, robot_quat, robot_anchor_pos, robot_anchor_quat)
    ref_local_pos = local_positions(ref_pos, ref_quat, ref_anchor_pos, ref_anchor_quat)
    local_errors = np.linalg.norm(robot_local_pos - ref_local_pos, axis=-1)
    ref_joint_pos = motion_loader.joint_pos[step_idx][dm.isaaclab_to_mujoco_reindex]
    dof_position_error = np.abs(sim_data.qpos[7:] - ref_joint_pos)
    joint_torque_abs = np.abs(np.asarray(joint_torque, dtype=np.float64))

    values = {
        "global_mpjpe": float(global_errors.mean()),
        "local_mpjpe": float(local_errors.mean()),
        "anchor_global_pos_error": float(np.linalg.norm(robot_anchor_pos - ref_anchor_pos)),
        "anchor_local_pos_error": float(
            np.linalg.norm(
                dm.motion_anchor_pos_b_future(sim_data, motion_loader, step_idx).reshape(motion_loader.future_steps, 3)[0]
            )
        ),
        "mean_dof_position_error": float(dof_position_error.mean()),
        "mean_joint_torque": float(joint_torque_abs.mean()),
        "joint_torque_std": float(joint_torque_abs.std()),
        "_joint_torque_abs_sum": float(joint_torque_abs.sum()),
        "_joint_torque_abs_sq_sum": float(np.square(joint_torque_abs).sum()),
        "_joint_torque_count": float(joint_torque_abs.size),
    }

    eval_ref_indexes = [motion_loader.motion_body_index(name) for name in EVAL_BODY_NAMES]
    eval_robot_pos = np.stack([sim_data.body(name).xpos.copy() for name in EVAL_BODY_NAMES], axis=0)
    eval_robot_quat = np.stack([sim_data.body(name).xquat.copy() for name in EVAL_BODY_NAMES], axis=0)
    eval_ref_pos = motion_loader.body_pos[step_idx, eval_ref_indexes]
    eval_ref_quat = motion_loader.body_ori[step_idx, eval_ref_indexes]
    eval_pos_errors = np.linalg.norm(eval_robot_pos - eval_ref_pos, axis=-1)
    eval_ori_errors = quat_error_rad(eval_ref_quat, eval_robot_quat)
    for i, name in enumerate(EVAL_BODY_NAMES):
        values[f"{name}_global_pos_error"] = float(eval_pos_errors[i])
        values[f"{name}_global_ori_error_rad"] = float(eval_ori_errors[i])
    ee_pos_errors = [
        values[f"{name}_global_pos_error"]
        for name in EVAL_BODY_NAMES
        if name != "pelvis"
    ]
    values["max_ee_global_pos_error"] = float(max(ee_pos_errors))
    values["pelvis_global_height"] = float(eval_robot_pos[EVAL_BODY_NAMES.index("pelvis"), 2])
    return values


def evaluate_motion(
    motion_file: str,
    session,
    obs_name: str,
    policy_input_dim: int | None,
    task: str,
    observation_terms: tuple[str, ...],
    future_steps: int,
    anchor_body_name: str,
    body_names: list[str],
    args: argparse.Namespace,
    motion_ae_adapter,
    motion_transformer_vae_adapter,
) -> dict:
    xml_path = Path(args.xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = args.sim_dt

    motion_loader = dm.MotionLoader(
        motion_file,
        future_steps=future_steps,
        anchor_body_name=anchor_body_name,
        body_names=body_names,
        motion_ae_adapter=motion_ae_adapter,
        motion_transformer_vae_adapter=motion_transformer_vae_adapter,
    )

    action = np.zeros(dm.NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = motion_loader.joint_pos[0][dm.isaaclab_to_mujoco_reindex].astype(np.float32, copy=True)
    data.qpos[7:] = motion_loader.joint_pos[0][dm.isaaclab_to_mujoco_reindex]
    data.qpos[:3] = motion_loader.body_pos[0][motion_loader.anchor_body_index]
    data.qpos[3:7] = motion_loader.body_ori[0][motion_loader.anchor_body_index]
    mujoco.mj_forward(model, data)

    metrics = MetricAccumulator()
    termination = TerminationTracker(args)
    eval_indexes = [motion_loader.motion_body_index(name) for name in EVAL_BODY_NAMES]
    inner_counter = 0
    last_tau = np.zeros(dm.NUM_ACTIONS, dtype=np.float64)
    max_frames = motion_loader.T if args.max_frames <= 0 else min(args.max_frames, motion_loader.T)
    terminated = False
    failure_frame = None
    failure_reason = "none"
    while inner_counter < max_frames:
        for _ in range(args.control_decimation):
            tau = dm.pd_control(target_dof_pos, data.qpos[7:], dm.kps, np.zeros_like(dm.kds), data.qvel[6:], dm.kds)
            data.ctrl[:] = tau
            last_tau = tau.copy()
            mujoco.mj_step(model, data)

        obs = dm.compute_observation(
            data,
            motion_loader,
            inner_counter,
            action,
            obs_config=args.obs_config,
            task=task,
            observation_terms=observation_terms,
        )
        obs_tensor = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if policy_input_dim is not None and obs_tensor.shape[1] != policy_input_dim:
            raise RuntimeError(
                f"Computed obs dim {obs_tensor.shape[1]} does not match ONNX input dim {policy_input_dim}"
            )
        action = session.run(None, {obs_name: obs_tensor})[0].squeeze().astype(np.float32)
        target_dof_pos = action[dm.isaaclab_to_mujoco_reindex] * dm.action_scale + dm.default_angles
        frame_metrics = compute_frame_metrics(data, motion_loader, inner_counter, eval_indexes, last_tau)
        metrics.update(frame_metrics)
        failure = termination.update(frame_metrics, data)
        if failure is not None:
            terminated = True
            failure_frame = inner_counter
            failure_reason = failure
            inner_counter += 1
            break
        inner_counter += 1

    summary = metrics.summary()
    summary["motion_file"] = motion_file
    summary["motion_length"] = motion_loader.T
    summary["evaluated_frames"] = inner_counter
    summary["terminated"] = terminated
    summary["success"] = not terminated and inner_counter >= motion_loader.T
    summary["progress"] = min(float(inner_counter) / float(max(motion_loader.T, 1)), 1.0)
    summary["failure_frame"] = -1 if failure_frame is None else failure_frame
    summary["failure_reason"] = failure_reason
    summary["max_frames_limited"] = args.max_frames > 0 and args.max_frames < motion_loader.T
    return summary


def flatten_row(summary: dict) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "motion_file": summary["motion_file"],
        "frames": summary["frames"],
        "motion_length": summary["motion_length"],
        "evaluated_frames": summary["evaluated_frames"],
        "terminated": int(summary["terminated"]),
        "success": int(summary["success"]),
        "progress": summary["progress"],
        "failure_frame": summary["failure_frame"],
        "failure_reason": summary["failure_reason"],
        "max_frames_limited": int(summary["max_frames_limited"]),
        "global_mpjpe": summary["global_mpjpe"],
        "local_mpjpe": summary["local_mpjpe"],
        "anchor_global_pos_error": summary["anchor_global_pos_error"],
        "anchor_local_pos_error": summary["anchor_local_pos_error"],
        "mean_dof_position_error": summary["mean_dof_position_error"],
        "mean_joint_torque": summary["mean_joint_torque"],
        "joint_torque_std": summary["joint_torque_std"],
    }
    for group in ("eval_body_global_pos_error", "eval_body_global_ori_error_rad"):
        for name, value in summary[group].items():
            row[f"{name}_{group}"] = value
    return row


def public_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key not in INTERNAL_METRIC_KEYS}


def aggregate(summaries: list[dict]) -> dict:
    total_frames = sum(int(item["frames"]) for item in summaries)
    if total_frames <= 0:
        raise RuntimeError("No evaluated frames")

    success_items = [item for item in summaries if bool(item["success"])]

    def weighted_scalar(key: str, items: list[dict] | None = None) -> float:
        selected = summaries if items is None else items
        selected_frames = sum(int(item["frames"]) for item in selected)
        if selected_frames <= 0:
            return 0.0
        return float(sum(float(item[key]) * int(item["frames"]) for item in selected) / selected_frames)

    def torque_stats(items: list[dict]) -> tuple[float, float]:
        count = sum(int(item["_joint_torque_count"]) for item in items)
        if count <= 0:
            return 0.0, 0.0
        torque_sum = sum(float(item["_joint_torque_abs_sum"]) for item in items)
        torque_sq_sum = sum(float(item["_joint_torque_abs_sq_sum"]) for item in items)
        mean = torque_sum / count
        var = torque_sq_sum / count - mean**2
        return float(mean), float(np.sqrt(max(var, 0.0)))

    all_torque_mean, all_torque_std = torque_stats(summaries)
    success_torque_mean, success_torque_std = torque_stats(success_items)

    result: dict = {
        "num_motions": len(summaries),
        "frames": total_frames,
        "success_count": len(success_items),
        "terminated_count": sum(1 for item in summaries if bool(item["terminated"])),
        "success_rate": float(np.mean([bool(item["success"]) for item in summaries])),
        "progress_rate": float(np.mean([float(item["progress"]) for item in summaries])),
        "failure_reasons": {
            reason: sum(1 for item in summaries if item["failure_reason"] == reason)
            for reason in ("none", *FAIL_REASON_KEYS)
        },
        "all": {
            "global_mpjpe": weighted_scalar("global_mpjpe"),
            "local_mpjpe": weighted_scalar("local_mpjpe"),
            "anchor_global_pos_error": weighted_scalar("anchor_global_pos_error"),
            "anchor_local_pos_error": weighted_scalar("anchor_local_pos_error"),
            "mean_dof_position_error": weighted_scalar("mean_dof_position_error"),
            "mean_joint_torque": all_torque_mean,
            "joint_torque_std": all_torque_std,
            "eval_body_global_pos_error": {},
            "eval_body_global_ori_error_rad": {},
        },
        "success": {
            "global_mpjpe": weighted_scalar("global_mpjpe", success_items),
            "local_mpjpe": weighted_scalar("local_mpjpe", success_items),
            "anchor_global_pos_error": weighted_scalar("anchor_global_pos_error", success_items),
            "anchor_local_pos_error": weighted_scalar("anchor_local_pos_error", success_items),
            "mean_dof_position_error": weighted_scalar("mean_dof_position_error", success_items),
            "mean_joint_torque": success_torque_mean,
            "joint_torque_std": success_torque_std,
            "eval_body_global_pos_error": {},
            "eval_body_global_ori_error_rad": {},
        },
    }
    for subset_name, items in (("all", summaries), ("success", success_items)):
        selected_frames = sum(int(item["frames"]) for item in items)
        for group in ("eval_body_global_pos_error", "eval_body_global_ori_error_rad"):
            for name in EVAL_BODY_NAMES:
                if selected_frames <= 0:
                    result[subset_name][group][name] = 0.0
                else:
                    result[subset_name][group][name] = float(
                        sum(float(item[group][name]) * int(item["frames"]) for item in items) / selected_frames
                    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", required=True, help="Path to exported policy ONNX")
    parser.add_argument("--motion_folder", required=True, help="Folder, glob, or single motion.npz to evaluate")
    parser.add_argument("--output_dir", default="eval_mujoco_onnx", help="Directory for metrics JSON/CSV")
    parser.add_argument("--task", default=dm.TASK_AUTO, choices=(dm.TASK_AUTO, *dm.SUPPORTED_TASKS))
    parser.add_argument("--future_steps", type=int, default=None)
    parser.add_argument("--obs_config", default="ProjGravAnchorEEObs")
    parser.add_argument("--anchor_body_name", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N motions; 0 means all")
    parser.add_argument("--max_frames", type=int, default=0, help="Evaluate only first N frames per motion; 0 means full motion")
    parser.add_argument("--sim_dt", type=float, default=0.002)
    parser.add_argument("--control_decimation", type=int, default=10)
    parser.add_argument("--consecutive_fail_frames", type=int, default=10)
    parser.add_argument("--pelvis_height_min", type=float, default=0.35)
    parser.add_argument("--global_mpjpe_fail", type=float, default=0.35)
    parser.add_argument("--local_mpjpe_fail", type=float, default=0.25)
    parser.add_argument("--anchor_global_pos_fail", type=float, default=0.60)
    parser.add_argument("--ee_global_pos_fail", type=float, default=0.40)
    parser.add_argument(
        "--xml_path",
        default="./source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1_act.xml",
    )
    parser.add_argument("--motion_ae_encoder_onnx_path", default=dm.DEFAULT_MOTION_AE_ENCODER_ONNX_PATH)
    parser.add_argument("--motion_ae_project_root", default=dm.DEFAULT_MOTION_AE_PROJECT_ROOT)
    parser.add_argument("--motion_ae_config_path", default=dm.DEFAULT_MOTION_AE_CONFIG_PATH)
    parser.add_argument("--motion_ae_stats_path", default=dm.DEFAULT_MOTION_AE_STATS_PATH)
    parser.add_argument("--motion_ae_batch_size", type=int, default=4096)
    parser.add_argument("--motion_transformer_vae_encoder_onnx_path", default=dm.DEFAULT_MOTION_TRANSFORMER_VAE_ENCODER_ONNX_PATH)
    parser.add_argument("--motion_transformer_vae_project_root", default=dm.DEFAULT_MOTION_TRANSFORMER_VAE_PROJECT_ROOT)
    parser.add_argument("--motion_transformer_vae_config_path", default=dm.DEFAULT_MOTION_TRANSFORMER_VAE_CONFIG_PATH)
    parser.add_argument("--motion_transformer_vae_stats_path", default=dm.DEFAULT_MOTION_TRANSFORMER_VAE_STATS_PATH)
    parser.add_argument("--motion_transformer_vae_batch_size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    ort = dm.load_onnxruntime()
    session = ort.InferenceSession(str(Path(args.policy_path).expanduser()))
    obs_name = session.get_inputs()[0].name
    policy_metadata = dm.get_policy_metadata(session)
    policy_input_dim = dm.get_policy_input_dim(session)
    task = dm.resolve_task(
        args.task,
        policy_metadata=policy_metadata,
        obs_config=args.obs_config,
        policy_input_dim=policy_input_dim,
        future_steps=args.future_steps,
    )
    observation_terms = dm.get_observation_terms(task=task)
    future_steps = args.future_steps
    if future_steps is None:
        future_steps = dm.infer_future_steps_from_input_dim(
            policy_input_dim,
            obs_config=args.obs_config,
            task=task,
            observation_terms=observation_terms,
        )
    if future_steps is None:
        raise ValueError("Unable to infer future_steps. Pass --future_steps.")
    anchor_body_name = args.anchor_body_name or policy_metadata.get("anchor_body_name", "pelvis")
    body_names = dm._csv_metadata_list(policy_metadata.get("body_names")) or dm.DEFAULT_BODY_NAMES
    motion_files = resolve_motion_files(args.motion_folder, limit=args.limit)
    motion_ae_adapter, motion_transformer_vae_adapter = make_latent_adapters(task, args)

    started = time.time()
    summaries = []
    for index, motion_file in enumerate(motion_files, start=1):
        print(f"[{index}/{len(motion_files)}] evaluating {motion_file}")
        summaries.append(
            evaluate_motion(
                motion_file,
                session,
                obs_name,
                policy_input_dim,
                task,
                observation_terms,
                future_steps,
                anchor_body_name,
                body_names,
                args,
                motion_ae_adapter,
                motion_transformer_vae_adapter,
            )
        )

    aggregate_summary = aggregate(summaries)
    payload = {
        "policy_path": str(Path(args.policy_path).expanduser()),
        "motion_folder": args.motion_folder,
        "task": task,
        "future_steps": future_steps,
        "anchor_body_name": anchor_body_name,
        "observation_terms": list(observation_terms),
        "success_criteria": {
            "consecutive_fail_frames": max(int(args.consecutive_fail_frames), 1),
            "pelvis_height_min": args.pelvis_height_min,
            "global_mpjpe_fail": args.global_mpjpe_fail,
            "local_mpjpe_fail": args.local_mpjpe_fail,
            "anchor_global_pos_fail": args.anchor_global_pos_fail,
            "ee_global_pos_fail": args.ee_global_pos_fail,
        },
        "elapsed_sec": time.time() - started,
        "aggregate": aggregate_summary,
        "motions": [public_summary(item) for item in summaries],
    }
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "per_motion_metrics.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [flatten_row(item) for item in summaries]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(aggregate_summary, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
