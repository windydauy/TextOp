"""Batch convert KimoMotion numeric csv clips to motion.npz via MuJoCo FK.

KimoMotion csv clips use the legacy numeric layout consumed by
``kimomotion_csv_batch_to_npz.py``:

    root_pos(3), root_quat(4), G1 joint_pos(29)

This script keeps the manifest, per-clip output, stitched output, and
``segments.csv`` behavior from the IsaacLab batch converter, but uses MuJoCo
forward kinematics so it does not need to replay every clip in IsaacLab.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from motion_conversion_common import (
    clip_segments_to_frame_count,
    extract_motion_components,
    load_csv_motion_array,
    read_manifest_rows,
    stitch_csv_motion_clips,
    write_segments_csv,
)
from mujoco_csv_to_npz import (
    ISAAC_G1_BODY_NAMES,
    ISAAC_RUNTIME_JOINT_NAMES,
    MUJOCO_CSV_JOINT_NAMES,
    ConvertedMotion,
    _build_joint_name_to_addresses,
    _ensure_quaternion_continuity,
    _load_mujoco,
    _resolve_xml_path,
    _so3_derivative_wxyz,
    _write_frame_to_data,
    build_mujoco_body_name_to_id,
    compute_body_angular_velocities,
    compute_body_linear_velocities,
    load_mjcf_body_names,
    reorder_joint_series_to_runtime_order,
    resample_motion,
)


def _normalize_quat_series_wxyz(root_quat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        bad_frames = np.flatnonzero(norms[:, 0] == 0.0)
        raise ValueError(f"Root quaternion norm is zero at frames: {bad_frames[:10].tolist()}")
    return root_quat / norms


def _load_legacy_motion_components(
    motion: np.ndarray,
    root_quat_order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_pos, root_quat, joint_pos = extract_motion_components(
        motion,
        root_quat_order=root_quat_order,
    )
    if joint_pos.shape[1] != len(MUJOCO_CSV_JOINT_NAMES):
        raise ValueError(
            "KimoMotion csv must contain "
            f"{len(MUJOCO_CSV_JOINT_NAMES)} G1 joint columns after root pose, "
            f"got {joint_pos.shape[1]}."
        )
    return (
        np.asarray(root_pos, dtype=np.float64),
        _normalize_quat_series_wxyz(np.asarray(root_quat, dtype=np.float64)),
        np.asarray(joint_pos, dtype=np.float64),
    )


def convert_motion_array_to_npz(
    motion: np.ndarray,
    output_name: str | Path,
    robot_xml: str | Path,
    input_fps: int,
    output_fps: int,
    root_quat_order: str,
) -> ConvertedMotion:
    """Convert one KimoMotion numeric motion array with MuJoCo FK."""
    mujoco = _load_mujoco()
    root_pos_in, root_quat_in, joint_pos_in = _load_legacy_motion_components(
        motion,
        root_quat_order=root_quat_order,
    )
    root_pos, root_quat, joint_pos = resample_motion(
        root_pos_in,
        root_quat_in,
        joint_pos_in,
        input_fps=input_fps,
        output_fps=output_fps,
    )

    joint_vel = np.gradient(joint_pos, 1.0 / output_fps, axis=0)
    root_lin_vel = np.gradient(root_pos, 1.0 / output_fps, axis=0)
    root_ang_vel = _so3_derivative_wxyz(
        _ensure_quaternion_continuity(root_quat[:, None, :])[:, 0],
        1.0 / output_fps,
    )

    xml_path = _resolve_xml_path(robot_xml)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    mjcf_body_names = load_mjcf_body_names(xml_path)
    body_name_to_id = build_mujoco_body_name_to_id(mjcf_body_names)
    missing_bodies = [name for name in ISAAC_G1_BODY_NAMES if name not in body_name_to_id]
    if missing_bodies:
        raise KeyError(f"MuJoCo XML is missing Isaac body names: {missing_bodies}")

    joint_name_to_qpos_adr, joint_name_to_dof_adr = _build_joint_name_to_addresses(mujoco, model)
    missing_joints = [name for name in MUJOCO_CSV_JOINT_NAMES if name not in joint_name_to_qpos_adr]
    if missing_joints:
        raise KeyError(f"MuJoCo XML is missing KimoMotion G1 joints: {missing_joints}")

    num_frames = joint_pos.shape[0]
    body_pos_w = np.zeros((num_frames, len(ISAAC_G1_BODY_NAMES), 3), dtype=np.float64)
    body_quat_w = np.zeros((num_frames, len(ISAAC_G1_BODY_NAMES), 4), dtype=np.float64)

    for frame_idx in range(num_frames):
        _write_frame_to_data(
            data,
            joint_name_to_qpos_adr,
            joint_name_to_dof_adr,
            MUJOCO_CSV_JOINT_NAMES,
            root_pos[frame_idx],
            root_quat[frame_idx],
            root_lin_vel[frame_idx],
            root_ang_vel[frame_idx],
            joint_pos[frame_idx],
            joint_vel[frame_idx],
        )
        mujoco.mj_forward(model, data)
        for body_idx, body_name in enumerate(ISAAC_G1_BODY_NAMES):
            mujoco_body_id = body_name_to_id[body_name]
            body_pos_w[frame_idx, body_idx] = data.xpos[mujoco_body_id]
            body_quat_w[frame_idx, body_idx] = data.xquat[mujoco_body_id]

    body_lin_vel_w = compute_body_linear_velocities(body_pos_w, fps=output_fps)
    body_ang_vel_w = compute_body_angular_velocities(body_quat_w, fps=output_fps)
    runtime_joint_pos = reorder_joint_series_to_runtime_order(joint_pos)
    runtime_joint_vel = reorder_joint_series_to_runtime_order(joint_vel)

    output_path = Path(output_name)
    if output_path.suffix != ".npz":
        output_path = output_path / "motion.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=np.array([output_fps], dtype=np.int64),
        joint_pos=runtime_joint_pos.astype(np.float32),
        joint_vel=runtime_joint_vel.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        joint_names=np.array(ISAAC_RUNTIME_JOINT_NAMES, dtype="<U64"),
        body_names=np.array(ISAAC_G1_BODY_NAMES, dtype="<U64"),
    )
    return ConvertedMotion(
        output_path=output_path,
        fps=output_fps,
        root_pos=root_pos,
        root_quat=root_quat,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        joint_pos=runtime_joint_pos,
        joint_vel=runtime_joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )


def convert_single_csv(
    csv_path: str | Path,
    output_name: str | Path,
    robot_xml: str | Path,
    input_fps: int,
    output_fps: int,
    root_quat_order: str,
) -> ConvertedMotion:
    motion = load_csv_motion_array(csv_path)
    return convert_motion_array_to_npz(
        motion=motion,
        output_name=output_name,
        robot_xml=robot_xml,
        input_fps=input_fps,
        output_fps=output_fps,
        root_quat_order=root_quat_order,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch convert KimoMotion numeric csv clips to motion.npz via MuJoCo FK."
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to the KimoMotion manifest csv.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root used to resolve manifest csv_path.")
    parser.add_argument(
        "--per_clip_output_root",
        type=str,
        required=True,
        help="Directory where per-clip <clip_id>/motion.npz outputs will be written.",
    )
    parser.add_argument(
        "--stitched_output_root",
        type=str,
        required=True,
        help="Directory where stitched motion.npz and segments.csv will be written.",
    )
    parser.add_argument(
        "--robot_xml",
        type=str,
        default="TextOpTracker/source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1.xml",
        help="MuJoCo XML path used for forward kinematics.",
    )
    parser.add_argument("--buffer_seconds", type=float, default=0.2, help="Hold-buffer duration inserted between clips.")
    parser.add_argument(
        "--root_quat_order",
        type=str,
        default="wxyz",
        choices=("xyzw", "wxyz"),
        help="Quaternion order in KimoMotion root pose columns.",
    )
    parser.add_argument("--input_fps", type=int, default=50, help="Source KimoMotion csv fps.")
    parser.add_argument("--output_fps", type=int, default=50, help="Target motion.npz fps.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    dataset_root = Path(args.dataset_root)
    manifest_rows = read_manifest_rows(args.manifest)
    per_clip_output_root = Path(args.per_clip_output_root)
    stitched_output_root = Path(args.stitched_output_root)
    buffer_frames = int(round(args.buffer_seconds * args.input_fps))

    print(f"[INFO] Converting {len(manifest_rows)} KimoMotion clips via MuJoCo with buffer_frames={buffer_frames}")
    for index, row in enumerate(manifest_rows, start=1):
        csv_path = dataset_root / row["csv_path"]
        output_path = per_clip_output_root / row["clip_id"] / "motion.npz"
        print(f"[INFO] [{index}/{len(manifest_rows)}] Converting {row['clip_id']} -> {output_path}")
        convert_single_csv(
            csv_path=csv_path,
            output_name=output_path,
            robot_xml=args.robot_xml,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            root_quat_order=args.root_quat_order,
        )

    stitched_motion, raw_segments = stitch_csv_motion_clips(
        manifest_rows=manifest_rows,
        dataset_root=dataset_root,
        buffer_frames=buffer_frames,
    )
    stitched_motion_path = stitched_output_root / "motion.npz"
    print(f"[INFO] Converting stitched motion -> {stitched_motion_path}")
    stitched_converted = convert_motion_array_to_npz(
        motion=stitched_motion,
        output_name=stitched_motion_path,
        robot_xml=args.robot_xml,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        root_quat_order=args.root_quat_order,
    )

    output_segments = clip_segments_to_frame_count(
        raw_segments,
        frame_count=stitched_converted.joint_pos.shape[0],
    )
    write_segments_csv(output_segments, stitched_output_root / "segments.csv")
    print(f"[INFO] Wrote stitched segment metadata -> {stitched_output_root / 'segments.csv'}")


if __name__ == "__main__":
    main()
