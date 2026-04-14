"""Convert G1 header CSV motions to Isaac-compatible motion.npz using MuJoCo FK.

The generated NPZ keeps the same core keys as Isaac-side conversion:
    fps, joint_pos, joint_vel,
    body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

Additionally, the file stores metadata keys:
    joint_names, body_names

The body axis is exported in the Isaac runtime body order probed from
`textop_tracker.robots.g1.G1_CYLINDER_CFG`, so the result can be replayed by
IsaacLab without body-index scrambling.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation


MUJOCO_CSV_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ISAAC_RUNTIME_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


# Runtime-probed from IsaacLab on 2026-04-09 with G1_CYLINDER_CFG.
ISAAC_G1_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


@dataclass(frozen=True)
class ConvertedMotion:
    output_path: Path
    fps: int
    root_pos: np.ndarray
    root_quat: np.ndarray
    root_lin_vel: np.ndarray
    root_ang_vel: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray


def load_mjcf_body_names(xml_path: str | Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"No <worldbody> found in {xml_path}")
    names: list[str] = []

    def walk(node: ET.Element) -> None:
        for body in node.findall("body"):
            body_name = body.attrib.get("name")
            if body_name:
                names.append(body_name)
            walk(body)

    walk(worldbody)
    return names


def build_mujoco_body_name_to_id(body_names: Iterable[str]) -> dict[str, int]:
    # MuJoCo reserves body id 0 for the world body.
    return {name: i for i, name in enumerate(body_names, start=1)}


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("Quaternion norm is zero.")
    return quat / norm


def slerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat_wxyz(q0)
    q1 = _normalize_quat_wxyz(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return _normalize_quat_wxyz(out)
    theta_0 = math.acos(dot)
    theta = theta_0 * alpha
    q2 = _normalize_quat_wxyz(q1 - q0 * dot)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_mul_wxyz(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=np.float64,
    )


def _axis_angle_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat)
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = xyz / sin_half
    angle = 2.0 * math.atan2(sin_half, w)
    return axis * angle


def _so3_derivative_wxyz(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    if quats_wxyz.shape[0] < 2:
        return np.zeros((quats_wxyz.shape[0], 3), dtype=np.float64)
    q_prev = quats_wxyz[:-2]
    q_next = quats_wxyz[2:]
    omega = []
    for prev_q, next_q in zip(q_prev, q_next, strict=True):
        q_rel = _quat_mul_wxyz(next_q, _quat_conjugate_wxyz(prev_q))
        omega.append(_axis_angle_from_quat_wxyz(q_rel) / (2.0 * dt))
    omega_arr = np.stack(omega, axis=0)
    return np.concatenate([omega_arr[:1], omega_arr, omega_arr[-1:]], axis=0)


def _ensure_quaternion_continuity(quats_wxyz: np.ndarray) -> np.ndarray:
    quats = np.array(quats_wxyz, dtype=np.float64, copy=True)
    if quats.shape[0] < 2:
        return quats
    for i in range(1, quats.shape[0]):
        dots = np.sum(quats[i - 1] * quats[i], axis=-1)
        flip_mask = dots < 0.0
        quats[i, flip_mask] *= -1.0
    return quats


def compute_body_linear_velocities(body_pos_w: np.ndarray, fps: int) -> np.ndarray:
    return np.gradient(np.asarray(body_pos_w, dtype=np.float64), 1.0 / fps, axis=0)


def compute_body_angular_velocities(body_quat_w: np.ndarray, fps: int) -> np.ndarray:
    quats = _ensure_quaternion_continuity(np.asarray(body_quat_w, dtype=np.float64))
    num_frames, num_bodies, _ = quats.shape
    if num_frames < 2:
        return np.zeros((num_frames, num_bodies, 3), dtype=np.float64)

    angular_vel = np.zeros((num_frames, num_bodies, 3), dtype=np.float64)
    for body_idx in range(num_bodies):
        angular_vel[:, body_idx] = _so3_derivative_wxyz(quats[:, body_idx], 1.0 / fps)
    return angular_vel


_MUJOCO_TO_ISAAC_RUNTIME_REINDEX = [MUJOCO_CSV_JOINT_NAMES.index(name) for name in ISAAC_RUNTIME_JOINT_NAMES]


def reorder_joint_series_to_runtime_order(joint_series: np.ndarray) -> np.ndarray:
    joint_series = np.asarray(joint_series)
    return joint_series[:, _MUJOCO_TO_ISAAC_RUNTIME_REINDEX]


def _euler_xyz_deg_to_quat_wxyz(euler_deg: np.ndarray) -> np.ndarray:
    """(T, 3) extrinsic-XYZ degrees (SOMA convention) -> (T, 4) quat wxyz.

    Uses scipy lowercase 'xyz' = extrinsic rotations about world-fixed axes,
    which matches the convention used by SOMA and the reference viser scripts.
    """
    rad = np.deg2rad(euler_deg)
    q_xyzw = _Rotation.from_euler("xyz", rad).as_quat()  # scipy xyzw
    quats = np.column_stack([q_xyzw[:, 3], q_xyzw[:, :3]])  # -> wxyz
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


def load_header_csv_motion(
    csv_path: str | Path,
    joint_names: list[str],
    translation_scale: float,
    angles_in_degrees: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Path(csv_path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    def col(name: str) -> np.ndarray:
        return np.array([float(row[name]) for row in rows], dtype=np.float64)

    root_pos = np.stack(
        [col("root_translateX"), col("root_translateY"), col("root_translateZ")],
        axis=1,
    )
    root_pos = root_pos / float(translation_scale)

    root_euler = np.stack(
        [col("root_rotateX"), col("root_rotateY"), col("root_rotateZ")],
        axis=1,
    )
    if not angles_in_degrees:
        root_euler = np.rad2deg(root_euler)
    root_quat = _euler_xyz_deg_to_quat_wxyz(root_euler)

    dof = np.stack([col(f"{joint}_dof") for joint in joint_names], axis=1)
    if angles_in_degrees:
        dof = np.deg2rad(dof)

    return root_pos, root_quat, dof


def resample_motion(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    input_fps: int,
    output_fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if root_pos.shape[0] != root_quat.shape[0] or root_pos.shape[0] != joint_pos.shape[0]:
        raise ValueError("Input motion arrays must have the same frame count.")
    if root_pos.shape[0] < 2:
        raise ValueError("Need at least two frames for resampling.")

    input_dt = 1.0 / input_fps
    duration = (root_pos.shape[0] - 1) * input_dt
    output_dt = 1.0 / output_fps
    out_times = np.arange(0.0, duration + 1e-9, output_dt, dtype=np.float64)
    in_times = np.linspace(0.0, duration, root_pos.shape[0], dtype=np.float64)

    pos_out = np.column_stack([np.interp(out_times, in_times, root_pos[:, i]) for i in range(3)])
    joint_out = np.column_stack(
        [np.interp(out_times, in_times, joint_pos[:, i]) for i in range(joint_pos.shape[1])]
    )

    quat_out = np.zeros((len(out_times), 4), dtype=np.float64)
    for i, t in enumerate(out_times):
        if t >= duration:
            quat_out[i] = root_quat[-1]
            continue
        frame = np.searchsorted(in_times, t, side="right") - 1
        frame = max(0, min(frame, len(in_times) - 2))
        t0 = in_times[frame]
        t1 = in_times[frame + 1]
        alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        quat_out[i] = slerp_quat_wxyz(root_quat[frame], root_quat[frame + 1], float(alpha))
    return pos_out, quat_out, joint_out


def _world_body_velocities(model, data) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    v = np.zeros((model.nbody, 6), dtype=np.float64)
    for body_id in range(model.nbody):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, v[body_id], 0)
    return v[:, 3:6], v[:, 0:3]


def _load_mujoco():
    import mujoco
    import mujoco.viewer

    return mujoco


def _resolve_xml_path(robot_xml: str | Path) -> Path:
    path = Path(robot_xml)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / path).resolve()


def _build_joint_name_to_addresses(mujoco, model) -> tuple[dict[str, int], dict[str, int]]:
    joint_name_to_qpos_adr: dict[str, int] = {}
    joint_name_to_dof_adr: dict[str, int] = {}
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None or model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint_name_to_qpos_adr[joint_name] = int(model.jnt_qposadr[joint_id])
        joint_name_to_dof_adr[joint_name] = int(model.jnt_dofadr[joint_id])
    return joint_name_to_qpos_adr, joint_name_to_dof_adr


def _write_frame_to_data(
    data,
    joint_name_to_qpos_adr: dict[str, int],
    joint_name_to_dof_adr: dict[str, int],
    joint_names: list[str],
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    data.qvel[:3] = root_lin_vel
    data.qvel[3:6] = root_ang_vel
    for joint_idx, joint_name in enumerate(joint_names):
        data.qpos[joint_name_to_qpos_adr[joint_name]] = joint_pos[joint_idx]
        data.qvel[joint_name_to_dof_adr[joint_name]] = joint_vel[joint_idx]


def convert_csv_to_npz(
    input_file: str | Path,
    output_name: str | Path,
    robot_xml: str | Path,
    input_fps: int,
    output_fps: int,
    translation_scale: float,
    angles_in_degrees: bool,
) -> ConvertedMotion:
    mujoco = _load_mujoco()

    root_pos_in, root_quat_in, joint_pos_in = load_header_csv_motion(
        input_file,
        MUJOCO_CSV_JOINT_NAMES,
        translation_scale=translation_scale,
        angles_in_degrees=angles_in_degrees,
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
    root_ang_vel = _so3_derivative_wxyz(_ensure_quaternion_continuity(root_quat[:, None, :])[:, 0], 1.0 / output_fps)

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
        raise KeyError(f"MuJoCo XML is missing Isaac joints: {missing_joints}")

    T = joint_pos.shape[0]
    body_pos_w = np.zeros((T, len(ISAAC_G1_BODY_NAMES), 3), dtype=np.float64)
    body_quat_w = np.zeros((T, len(ISAAC_G1_BODY_NAMES), 4), dtype=np.float64)

    for t in range(T):
        _write_frame_to_data(
            data,
            joint_name_to_qpos_adr,
            joint_name_to_dof_adr,
            MUJOCO_CSV_JOINT_NAMES,
            root_pos[t],
            root_quat[t],
            root_lin_vel[t],
            root_ang_vel[t],
            joint_pos[t],
            joint_vel[t],
        )

        mujoco.mj_forward(model, data)
        for i, body_name in enumerate(ISAAC_G1_BODY_NAMES):
            body_id = body_name_to_id[body_name]
            body_pos_w[t, i] = data.xpos[body_id]
            body_quat_w[t, i] = data.xquat[body_id]

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


def visualize_motion(
    converted_motion: ConvertedMotion,
    robot_xml: str | Path,
    joint_names: list[str] | None = None,
) -> None:
    mujoco = _load_mujoco()
    joint_names = joint_names or ISAAC_RUNTIME_JOINT_NAMES
    xml_path = _resolve_xml_path(robot_xml)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_name_to_qpos_adr, joint_name_to_dof_adr = _build_joint_name_to_addresses(mujoco, model)

    frame_dt = 1.0 / converted_motion.fps
    next_frame_time = time.perf_counter()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = converted_motion.root_pos[0]
        viewer.cam.lookat[2] += 0.6
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        for frame_idx in range(converted_motion.joint_pos.shape[0]):
            if hasattr(viewer, "is_running"):
                is_running = viewer.is_running() if callable(viewer.is_running) else viewer.is_running
                if not is_running:
                    break

            _write_frame_to_data(
                data,
                joint_name_to_qpos_adr,
                joint_name_to_dof_adr,
                joint_names,
                converted_motion.root_pos[frame_idx],
                converted_motion.root_quat[frame_idx],
                converted_motion.root_lin_vel[frame_idx],
                converted_motion.root_ang_vel[frame_idx],
                converted_motion.joint_pos[frame_idx],
                converted_motion.joint_vel[frame_idx],
            )
            mujoco.mj_forward(model, data)
            viewer.sync()

            next_frame_time += frame_dt
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert G1 CSV to Isaac-compatible motion.npz via MuJoCo FK.")
    parser.add_argument("--input_file", type=str, required=True, help="Header CSV file to convert.")
    parser.add_argument("--output_name", type=str, required=True, help="Output .npz path or artifact directory.")
    parser.add_argument(
        "--robot_xml",
        type=str,
        default="TextOpTracker/source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1.xml",
        help="MuJoCo XML path used for forward kinematics.",
    )
    parser.add_argument("--input_fps", type=int, default=120, help="Source CSV fps.")
    parser.add_argument("--output_fps", type=int, default=50, help="Target output fps.")
    parser.add_argument(
        "--translation_scale",
        type=float,
        default=100.0,
        help="Input translation unit scale. 100 means CSV root_translate is in centimetres.",
    )
    parser.add_argument(
        "--angles_in_degrees",
        action="store_true",
        default=True,
        help="Interpret root euler and *_dof columns as degrees.",
    )
    parser.add_argument(
        "--angles_in_radians",
        action="store_true",
        help="Interpret root euler and *_dof columns as radians.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Only save motion.npz without opening the MuJoCo playback window.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    angles_in_degrees = not args.angles_in_radians
    converted_motion = convert_csv_to_npz(
        input_file=args.input_file,
        output_name=args.output_name,
        robot_xml=args.robot_xml,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        translation_scale=args.translation_scale,
        angles_in_degrees=angles_in_degrees,
    )
    print(f"[DONE] saved {converted_motion.output_path}")
    if not args.headless:
        visualize_motion(converted_motion, args.robot_xml)


if __name__ == "__main__":
    main()
