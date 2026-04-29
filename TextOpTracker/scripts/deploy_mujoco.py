import re
import time
import mujoco, mujoco_viewer, mujoco.viewer
# `$ pip install mujoco-python-viewer` If `mujoco_viewer` not found.
import numpy as np
import onnxruntime as ort
import argparse
from enum import Enum
from pathlib import Path


# 定义枚举类型
class AnchorBody(Enum):
    PELVIS = 0
    TORSO_LINK = 9


anchor_body = AnchorBody.PELVIS


def resolve_anchor_body(anchor_body_name: str) -> AnchorBody:
    normalized_name = anchor_body_name.strip().upper()
    try:
        return AnchorBody[normalized_name]
    except KeyError as exc:
        valid_names = [member.name.lower() for member in AnchorBody]
        raise ValueError(
            f"Unsupported anchor_body_name: {anchor_body_name}. Expected one of {valid_names}."
        ) from exc

isaaclab_joint_names = [
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

mujoco_joint_names = [
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

stiffness_dict = {
    ".*_hip_pitch_joint": 40.17923847137318,
    ".*_hip_roll_joint": 99.09842777666113,
    ".*_hip_yaw_joint": 40.17923847137318,
    ".*_knee_joint": 99.09842777666113,
    ".*_ankle_pitch_joint": 28.50124619574858,
    ".*_ankle_roll_joint": 28.50124619574858,
    "waist_roll_joint": 28.50124619574858,
    "waist_pitch_joint": 28.50124619574858,
    "waist_yaw_joint": 40.17923847137318,
    ".*_shoulder_pitch_joint": 14.25062309787429,
    ".*_shoulder_roll_joint": 14.25062309787429,
    ".*_shoulder_yaw_joint": 14.25062309787429,
    ".*_elbow_joint": 14.25062309787429,
    ".*_wrist_roll_joint": 14.25062309787429,
    ".*_wrist_pitch_joint": 16.77832748089279,
    ".*_wrist_yaw_joint": 16.77832748089279,
}

damping_dict = {
    ".*_hip_pitch_joint": 2.5578897650279457,
    ".*_hip_roll_joint": 6.3088018534966395,
    ".*_hip_yaw_joint": 2.5578897650279457,
    ".*_knee_joint": 6.3088018534966395,
    ".*_ankle_pitch_joint": 1.814445686584846,
    ".*_ankle_roll_joint": 1.814445686584846,
    "waist_roll_joint": 1.814445686584846,
    "waist_pitch_joint": 1.814445686584846,
    "waist_yaw_joint": 2.5578897650279457,
    ".*_shoulder_pitch_joint": 0.907222843292423,
    ".*_shoulder_roll_joint": 0.907222843292423,
    ".*_shoulder_yaw_joint": 0.907222843292423,
    ".*_elbow_joint": 0.907222843292423,
    ".*_wrist_roll_joint": 0.907222843292423,
    ".*_wrist_pitch_joint": 1.06814150219,
    ".*_wrist_yaw_joint": 1.06814150219,
}
scale_dict = {
    ".*_hip_yaw_joint": 0.5475464652142303,
    ".*_hip_roll_joint": 0.3506614663788243,
    ".*_hip_pitch_joint": 0.5475464652142303,
    ".*_knee_joint": 0.3506614663788243,
    ".*_ankle_pitch_joint": 0.43857731392336724,
    ".*_ankle_roll_joint": 0.43857731392336724,
    "waist_roll_joint": 0.43857731392336724,
    "waist_pitch_joint": 0.43857731392336724,
    "waist_yaw_joint": 0.5475464652142303,
    ".*_shoulder_pitch_joint": 0.43857731392336724,
    ".*_shoulder_roll_joint": 0.43857731392336724,
    ".*_shoulder_yaw_joint": 0.43857731392336724,
    ".*_elbow_joint": 0.43857731392336724,
    ".*_wrist_roll_joint": 0.43857731392336724,
    ".*_wrist_pitch_joint": 0.07450087032950714,
    ".*_wrist_yaw_joint": 0.07450087032950714,
}


def get_param(joint_name, param_dict):
    for pattern, value in param_dict.items():
        if pattern.startswith(".*"):
            if joint_name.startswith("left") or joint_name.startswith("right"):
                if joint_name.endswith(pattern[3:]):
                    return value
        else:
            if joint_name == pattern:
                return value
    raise ValueError(f"No value found for joint: {joint_name}")


def get_action_scale(joint_name):
    for pattern, scale in scale_dict.items():
        # 用 left/right 替换 .*
        if pattern.startswith(".*"):
            if joint_name.startswith("left") or joint_name.startswith("right"):
                if joint_name.endswith(pattern[3:]):
                    return scale
        else:
            if joint_name == pattern:
                return scale
    raise ValueError(f"No scale found for joint: {joint_name}")


# 根据 isaaclab joint_pos 配置设置默认关节位置
joint_pos_config = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def get_joint_default_pos(joint_name):
    for pattern, pos in joint_pos_config.items():
        if pattern.startswith(".*"):
            if joint_name.startswith("left") or joint_name.startswith("right"):
                if joint_name.endswith(pattern[3:]):
                    return pos
        else:
            if joint_name == pattern:
                return pos
    return 0.0  # 默认值为0


isaaclab_to_mujoco_reindex = [isaaclab_joint_names.index(name) for name in mujoco_joint_names]
mujoco_to_isaaclab_reindex = [mujoco_joint_names.index(name) for name in isaaclab_joint_names]
joint_names = mujoco_joint_names
kps = [get_param(name, stiffness_dict) for name in joint_names]
kds = [get_param(name, damping_dict) for name in joint_names]
kps = np.array(kps, dtype=np.float32)
kds = np.array(kds, dtype=np.float32)
action_scale = np.array([get_action_scale(name) for name in joint_names], dtype=np.float32)
default_angles = np.array([get_joint_default_pos(name) for name in mujoco_joint_names], dtype=np.float32)

print("isaaclab_to_mujoco_reindex")
print(isaaclab_to_mujoco_reindex)
print("mujoco_to_isaaclab_reindex")
print(mujoco_to_isaaclab_reindex)

print("kp:")
print(kps)
print("kd:")
print(kds)

print("action_scale")
print(action_scale)

print("default joint pos:")
print(default_angles)


# Fix bug for mujoco_viewer:
def __fix__add_marker_to_scene(self, marker):
    if self.scn.ngeom >= self.scn.maxgeom:
        raise RuntimeError("Ran out of geoms. maxgeom: %d" % self.scn.maxgeom)

    g = self.scn.geoms[self.scn.ngeom]
    # default values.
    g.dataid = -1
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid = -1
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    # g.texid = -1
    # g.texuniform = 0
    # g.texrepeat[0] = 1
    # g.texrepeat[1] = 1
    g.emission = 0
    g.specular = 0.5
    g.shininess = 0.5
    g.reflectance = 0
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size[:] = np.ones(3) * 0.1
    g.mat[:] = np.eye(3)
    g.rgba[:] = np.ones(4)

    for key, value in marker.items():
        if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
            setattr(g, key, value)
        elif isinstance(value, (tuple, list, np.ndarray)):
            attr = getattr(g, key)
            attr[:] = np.asarray(value).reshape(attr.shape)
        elif isinstance(value, str):
            assert key == "label", "Only label is a string in mjtGeom."
            if value is None:
                g.label[0] = 0
            else:
                g.label = value
        elif hasattr(g, key):
            raise ValueError("mjtGeom has attr {} but type {} is invalid".format(key, type(value)))
        else:
            raise ValueError("mjtGeom doesn't have field %s" % key)

    self.scn.ngeom += 1

    return


mujoco_viewer.MujocoViewer._add_marker_to_scene = __fix__add_marker_to_scene


def update_joint_visualization(viewer, motion_loader, t):
    """Update joint visualization spheres using body positions"""
    if t < 0 or t >= motion_loader.T:
        return

    # Get body positions from motion data
    body_pos = motion_loader.body_pos[t]  # [N, 3]

    # Update spheres for each body (excluding the first one which is pelvis)
    for i in range(0, len(body_pos)):
        viewer.add_marker(
            pos=body_pos[i],
            size=[0.02, 0.02, 0.02],
            rgba=[0.8, 0.5, 0.3, 1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            label=""
        )

    # Update anchor body sphere (larger) - use torso_link position
    anchor_pos = body_pos[motion_loader.anchor_body_index]  # torso_link position
    viewer.add_marker(
        pos=anchor_pos, size=[0.05, 0.05, 0.05], rgba=[0.8, 0.5, 0.3, 1], type=mujoco.mjtGeom.mjGEOM_SPHERE, label=""
    )


# ====== MotionLoader 类（参考 isaaclab） ======
class MotionLoader:
    def __init__(self, motion_file, anchor_body_name: str | None = None):
        p = Path(motion_file)
        motion_path = str(p / "motion.npz") if p.is_dir() else str(p)
        print(motion_path)
        data = np.load(motion_path)
        self.joint_pos = data["joint_pos"]  # [T, 29]
        self.joint_vel = data["joint_vel"]  # [T, 29]
        self.body_pos = data["body_pos_w"]  # [T, N, 3]
        self.body_ori = data["body_quat_w"]  # [T, N, 4]
        # self.body_ang_vel_w = data["body_ang_vel_w"]
        self.fps = data["fps"]
        self.T = self.joint_pos.shape[0]

        # self.body_pos[..., 2] += 0.015

        # G1 body names from the config
        self.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        selected_anchor_body = resolve_anchor_body(anchor_body_name) if anchor_body_name is not None else anchor_body
        self.anchor_body_name = selected_anchor_body.name.lower()
        self.anchor_body_index = self.body_names.index(self.anchor_body_name)

        # Future steps configuration
        self.future_steps = 5


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v (NumPy version).

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]

    # Component a: v * (2.0 * q_w^2 - 1.0)
    a = v * np.expand_dims(2.0 * q_w**2 - 1.0, axis=-1)

    # Component b: cross(q_vec, v) * q_w * 2.0
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0

    # Component c: q_vec * dot(q_vec, v) * 2.0
    # For efficient computation, handle different dimensionalities
    if q_vec.ndim == 2:
        # For 2D case: use matrix multiplication for better performance
        dot_product = np.sum(q_vec * v, axis=-1, keepdims=True)
        c = q_vec * dot_product * 2.0
    else:
        # For general case: use Einstein summation
        dot_product = np.expand_dims(np.einsum('...i,...i->...', q_vec, v), axis=-1)
        c = q_vec * dot_product * 2.0

    return a - b + c


def _quat_mul_wxyz_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product, both quaternions wxyz. Broadcasts on leading dims."""
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rw, rx, ry, rz = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    out = np.empty(np.broadcast_shapes(q.shape, r.shape), dtype=np.float64)
    out[..., 0] = qw * rw - qx * rx - qy * ry - qz * rz
    out[..., 1] = qw * rx + qx * rw + qy * rz - qz * ry
    out[..., 2] = qw * ry - qx * rz + qy * rw + qz * rx
    out[..., 3] = qw * rz + qx * ry - qy * rx + qz * rw
    return out


def _quat_inv_wxyz_np(q: np.ndarray) -> np.ndarray:
    """Unit quaternion inverse (conjugate) wxyz."""
    q = np.asarray(q, dtype=np.float64)
    out = np.empty_like(q)
    out[..., 0] = q[..., 0]
    out[..., 1:] = -q[..., 1:]
    return out


def matrix_from_quat_np(quat: np.ndarray) -> np.ndarray:
    """wxyz -> rotation matrix R (world from body), shape (..., 3, 3)."""
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)
    R = np.stack(
        [
            np.stack([m00, m01, m02], axis=-1),
            np.stack([m10, m11, m12], axis=-1),
            np.stack([m20, m21, m22], axis=-1),
        ],
        axis=-2,
    )
    return R.astype(np.float32, copy=False)


def subtract_frame_transforms_np(
    t01: np.ndarray,
    q01: np.ndarray,
    t02: np.ndarray,
    q02: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match Isaac Lab: relative pose of (t02,q02) expressed in frame of (t01,q01); world-frame inputs."""
    t01 = np.asarray(t01, dtype=np.float64)
    q01 = np.asarray(q01, dtype=np.float64)
    t02 = np.asarray(t02, dtype=np.float64)
    q02 = np.asarray(q02, dtype=np.float64)
    R_w_1 = matrix_from_quat_np(q01)
    if R_w_1.ndim == 2:
        delta = t02 - t01
        pos_b = (R_w_1.T @ delta.reshape(3, 1)).reshape(3)
    else:
        delta = t02 - t01
        R_1_w = np.swapaxes(R_w_1, -1, -2)
        pos_b = np.einsum("...ij,...j->...i", R_1_w, delta)
    ori_b = _quat_mul_wxyz_np(_quat_inv_wxyz_np(q01), q02).astype(np.float32, copy=False)
    pos_b = pos_b.astype(np.float32, copy=False)
    return pos_b, ori_b


# ====== observation 计算函数 ======
def get_command(motion_loader, t):
    """Get command (joint_pos + joint_vel) for future steps - matching Isaac Lab order"""
    if t < 0:
        return np.zeros(290, dtype=np.float32)  # 29 * 2 * 5 = 290

    # Get future joint positions and velocities - batch process like Isaac Lab
    future_joint_pos = []
    future_joint_vel = []

    for i in range(motion_loader.future_steps):
        step_idx = min(t + i, motion_loader.T - 1)
        future_joint_pos.append(motion_loader.joint_pos[step_idx])
        future_joint_vel.append(motion_loader.joint_vel[step_idx])

    # Stack to create [future_steps, 29] then flatten to [future_steps * 29]
    joint_pos_future = np.stack(future_joint_pos, axis=0).flatten()  # [29 * 5]
    joint_vel_future = np.stack(future_joint_vel, axis=0).flatten()  # [29 * 5]
    # breakpoint()

    cmd = np.concatenate([joint_pos_future, joint_vel_future], axis=0)
    # breakpoint()
    return cmd  # [290]


def motion_anchor_pos_b_future(sim_data, motion_loader, t):
    """Future N-step motion anchor position in body frame - matching Isaac Lab order"""
    if t < 0:
        return np.zeros(15, dtype=np.float32)  # 3 * 5 = 15

    # Get robot anchor pose
    robot_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy().reshape(1, 3)
    robot_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 4)

    # Get future motion anchor poses - batch process like Isaac Lab
    future_positions = []
    future_orientations = []
    for i in range(motion_loader.future_steps):
        step_idx = min(t + i, motion_loader.T - 1)
        ref_pos = motion_loader.body_pos[step_idx][motion_loader.anchor_body_index].reshape(1, 3)
        ref_quat = motion_loader.body_ori[step_idx][motion_loader.anchor_body_index].reshape(1, 4)
        future_positions.append(ref_pos)
        future_orientations.append(ref_quat)

    # Stack to create [future_steps, 1, 3] then reshape to [future_steps, 3]
    future_anchor_pos_w = np.stack(future_positions, axis=0).squeeze(1)  # [5, 3]
    future_anchor_quat_w = np.stack(future_orientations, axis=0).squeeze(1)  # [5, 4]

    # Expand robot anchor for broadcasting: [future_steps, 3] and [future_steps, 4]
    robot_anchor_pos_w_exp = robot_pos.repeat(motion_loader.future_steps, axis=0)  # [5, 3]
    robot_anchor_quat_w_exp = robot_quat.repeat(motion_loader.future_steps, axis=0)  # [5, 4]

    # Transform all future steps at once
    pos_b, _ = subtract_frame_transforms_np(
        robot_anchor_pos_w_exp,
        robot_anchor_quat_w_exp,
        future_anchor_pos_w,
        future_anchor_quat_w,
    )

    pos_b_flat = pos_b.flatten()  # [15]
    return pos_b_flat  # [15]


def motion_anchor_ori_b_future(sim_data, motion_loader, t):
    """Future N-step motion anchor orientation in body frame - matching Isaac Lab order"""
    if t < 0:
        return np.zeros(30, dtype=np.float32)  # 6 * 5 = 30

    # Get robot anchor pose
    robot_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy().reshape(1, 3)
    robot_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 4)

    # Get future motion anchor orientations - batch process like Isaac Lab
    future_positions = []
    future_orientations = []
    for i in range(motion_loader.future_steps):
        step_idx = min(t + i, motion_loader.T - 1)
        ref_pos = motion_loader.body_pos[step_idx][motion_loader.anchor_body_index].reshape(1, 3)
        ref_quat = motion_loader.body_ori[step_idx][motion_loader.anchor_body_index].reshape(1, 4)
        future_positions.append(ref_pos)
        future_orientations.append(ref_quat)
    # Stack to create [future_steps, 1, 3] then reshape to [future_steps, 3]
    future_anchor_pos_w = np.stack(future_positions, axis=0).squeeze(1)  # [5, 3]
    future_anchor_quat_w = np.stack(future_orientations, axis=0).squeeze(1)  # [5, 4]

    # Expand robot anchor for broadcasting: [future_steps, 3] and [future_steps, 4]
    robot_anchor_pos_w_exp = robot_pos.repeat(motion_loader.future_steps, axis=0)  # [5, 3]
    robot_anchor_quat_w_exp = robot_quat.repeat(motion_loader.future_steps, axis=0)  # [5, 4]

    # Transform all future steps at once
    pos_b, ori_b = subtract_frame_transforms_np(
        robot_anchor_pos_w_exp,
        robot_anchor_quat_w_exp,
        future_anchor_pos_w,
        future_anchor_quat_w,
    )

    # Convert to rotation matrix and take first 2 rows
    mat = matrix_from_quat_np(ori_b)
    mat_flat = mat[..., :2].reshape(-1)
    # breakpoint()
    ori_b_flat = np.array(mat_flat, dtype=np.float32).flatten()  # [30]
    # breakpoint()
    return ori_b_flat  # [30]


def robot_body_pos_b(sim_data, motion_loader, t):
    """Robot body positions in body frame"""
    # Get robot anchor pose
    robot_anchor_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy().reshape(1, 3)
    robot_anchor_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 4)

    # Get all robot body poses
    robot_body_positions = []
    robot_body_orientations = []

    for body_name in motion_loader.body_names:
        body_pos = sim_data.body(body_name).xpos.copy().reshape(1, 3)
        body_quat = sim_data.body(body_name).xquat.copy().reshape(1, 4)
        robot_body_positions.append(body_pos)
        robot_body_orientations.append(body_quat)

    robot_body_pos = np.concatenate(robot_body_positions, axis=0)  # [14, 3]
    robot_body_quat = np.concatenate(robot_body_orientations, axis=0)  # [14, 4]

    # Transform to body frame - expand robot_anchor to match robot_body shape
    robot_anchor_pos_expanded = robot_anchor_pos.repeat(len(motion_loader.body_names), axis=0)  # [14, 3]
    robot_anchor_quat_expanded = robot_anchor_quat.repeat(len(motion_loader.body_names), axis=0)  # [14, 4]

    pos_b, _ = subtract_frame_transforms_np(
        robot_anchor_pos_expanded,
        robot_anchor_quat_expanded,
        robot_body_pos,
        robot_body_quat,
    )

    return pos_b.flatten()  # [42]


def robot_body_ori_b(sim_data, motion_loader, t):
    """Robot body orientations in body frame"""
    # Get robot anchor pose
    robot_anchor_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy().reshape(1, 3)
    robot_anchor_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 4)

    # Get all robot body poses
    robot_body_positions = []
    robot_body_orientations = []

    for body_name in motion_loader.body_names:
        body_pos = sim_data.body(body_name).xpos.copy().reshape(1, 3)
        body_quat = sim_data.body(body_name).xquat.copy().reshape(1, 4)
        robot_body_positions.append(body_pos)
        robot_body_orientations.append(body_quat)

    robot_body_pos = np.concatenate(robot_body_positions, axis=0)  # [14, 3]
    robot_body_quat = np.concatenate(robot_body_orientations, axis=0)  # [14, 4]

    # Transform to body frame - expand robot_anchor to match robot_body shape
    robot_anchor_pos_expanded = robot_anchor_pos.repeat(len(motion_loader.body_names), axis=0)  # [14, 3]
    robot_anchor_quat_expanded = robot_anchor_quat.repeat(len(motion_loader.body_names), axis=0)  # [14, 4]

    _, ori_b = subtract_frame_transforms_np(
        robot_anchor_pos_expanded,
        robot_anchor_quat_expanded,
        robot_body_pos,
        robot_body_quat,
    )

    # Convert to rotation matrix and take first 2 rows
    mat = matrix_from_quat_np(ori_b)
    mat_flat = mat[..., :2].reshape(-1)

    return mat_flat.astype(np.float32, copy=False).flatten()  # [84]


def get_base_lin_vel(sim_data):
    """Get base linear velocity"""
    linvel_b = quat_rotate_inverse_np(sim_data.qpos[3:7], sim_data.qvel[0:3])
    # return sim_data.qvel[0:3]  # shape [3]
    return linvel_b


def get_base_ang_vel(sim_data):
    """Get base angular velocity
    
    Pay special attention: 
    - The angular velocity sim_data.qvel[3:6] is in the body frame, not the world frame.
    - The angular velocity sim_data.qvel[3:6] is in the body frame, not the world frame.
    """

    return sim_data.qvel[3:6]  # shape [3]


def get_joint_pos_rel(sim_data):
    """Get relative joint positions"""
    return sim_data.qpos[7:36] - default_angles  # shape [29]


def get_joint_vel_rel(sim_data):
    """Get relative joint velocities"""
    return sim_data.qvel[6:35]  # shape [29]


def get_last_action(last_actions):
    """Get last actions"""
    return last_actions  # shape [29]


def get_projected_gravity(sim_data):
    """Get projected gravity"""
    quaternion = sim_data.qpos[3:7]  # wxyz

    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def get_robot_anchor_pos_w(sim_data, motion_loader):
    """Robot anchor body position in world frame, shape (3,)."""
    return sim_data.body(motion_loader.anchor_body_name).xpos.copy().astype(np.float32)


def get_robot_anchor_ori_w(sim_data, motion_loader):
    """Robot anchor body orientation as first 2 rows of rotation matrix, shape (6,).

    MuJoCo xquat is (w, x, y, z), matching the wxyz convention used by matrix_from_quat_np.
    """
    quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy()  # wxyz
    mat = matrix_from_quat_np(quat)   # (3, 3)
    return mat[:2].flatten().astype(np.float32)  # (6,)


#


_OBS_DIMS = {
    # command(290) + anchor_pos_b(15) + anchor_ori_b(30) + proj_grav(3)
    # + lin_vel(3) + ang_vel(3) + joint_pos(29) + joint_vel(29) + actions(29)
    "ProjGravObs": 431,
    # same as above + robot_anchor_pos_w(3) + robot_anchor_ori_w(6)
    "ProjGravAnchorObs": 440,
}

_OBS_SEGMENTS = {
    "ProjGravObs": [
        ("command", 290),
        ("motion_anchor_pos_b", 15),
        ("motion_anchor_ori_b", 30),
        ("projected_gravity", 3),
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("joint_pos", 29),
        ("joint_vel", 29),
        ("actions", 29),
    ],
    "ProjGravAnchorObs": [
        ("command", 290),
        ("motion_anchor_pos_b", 15),
        ("motion_anchor_ori_b", 30),
        ("robot_anchor_pos_w", 3),
        ("robot_anchor_ori_w", 6),
        ("projected_gravity", 3),
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("joint_pos", 29),
        ("joint_vel", 29),
        ("actions", 29),
    ],
}


def iter_observation_segments(obs: np.ndarray, obs_config: str):
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    idx = 0
    for name, dim in _OBS_SEGMENTS[obs_config]:
        yield name, obs[idx:idx + dim]
        idx += dim
    if idx != obs.shape[0]:
        raise ValueError(
            f"Observation layout mismatch for {obs_config}: consumed {idx} values, got {obs.shape[0]}."
        )


def print_observation_debug(obs: np.ndarray, obs_config: str, t: int):
    print(f"\n[OBS DEBUG] t={t} obs_config={obs_config} dim={obs.shape[0]}")
    for name, segment in iter_observation_segments(obs, obs_config):
        values = np.array2string(
            segment,
            precision=5,
            separator=", ",
            threshold=segment.shape[0],
            max_line_width=100000,
        )
        print(f"  {name:<20} ({segment.shape[0]:>3d}) = {values}")
    print()


def compute_observation(sim_data, motion_loader, t, last_actions, obs_config: str = "ProjGravObs"):
    """Compute observation vector for the given obs_config.

    Supported configs:
      - "ProjGravObs"        : 431-dim  (Tracking-Flat-G1-ProjGravObs-MNMLP-v0)
      - "ProjGravAnchorObs"  : 440-dim  (Tracking-Flat-G1-ProjGravAnchorObs-NMMLP-v0
                                         and Tracking-Flat-G1-ProjGravAnchorObs-NMMLP5L-v0)
    """
    obs_dim = _OBS_DIMS[obs_config]

    if t < 0:
        return np.zeros(obs_dim, dtype=np.float32)

    obs = np.zeros(obs_dim, dtype=np.float32)
    idx = 0

    # 0. command (290,) - future joint pos + vel
    command = get_command(motion_loader, t)
    obs[idx:idx + command.shape[0]] = command
    idx += command.shape[0]

    # 1. motion_anchor_pos_b (15,) - future anchor pos in body frame
    motion_anchor_pos = motion_anchor_pos_b_future(sim_data, motion_loader, t)
    obs[idx:idx + motion_anchor_pos.shape[0]] = motion_anchor_pos
    idx += motion_anchor_pos.shape[0]

    # 2. motion_anchor_ori_b (30,) - future anchor ori in body frame
    motion_anchor_ori = motion_anchor_ori_b_future(sim_data, motion_loader, t)
    obs[idx:idx + motion_anchor_ori.shape[0]] = motion_anchor_ori
    idx += motion_anchor_ori.shape[0]

    # 3. [ProjGravAnchorObs only] robot_anchor_pos_w (3,) + robot_anchor_ori_w (6,)
    if obs_config == "ProjGravAnchorObs":
        robot_ap = get_robot_anchor_pos_w(sim_data, motion_loader)
        obs[idx:idx + robot_ap.shape[0]] = robot_ap
        idx += robot_ap.shape[0]

        robot_ao = get_robot_anchor_ori_w(sim_data, motion_loader)
        obs[idx:idx + robot_ao.shape[0]] = robot_ao
        idx += robot_ao.shape[0]

    # 4. projected_gravity (3,)
    projected_gravity = get_projected_gravity(sim_data)
    obs[idx:idx + projected_gravity.shape[0]] = projected_gravity
    idx += projected_gravity.shape[0]

    # 5. base_lin_vel (3,)
    base_lin_vel = get_base_lin_vel(sim_data)
    obs[idx:idx + base_lin_vel.shape[0]] = base_lin_vel
    idx += base_lin_vel.shape[0]

    # 6. base_ang_vel (3,)
    base_ang_vel = get_base_ang_vel(sim_data)
    obs[idx:idx + base_ang_vel.shape[0]] = base_ang_vel
    idx += base_ang_vel.shape[0]

    # 7. joint_pos (29,) - relative joint positions, reindexed to Isaac Lab order
    joint_pos = get_joint_pos_rel(sim_data)[mujoco_to_isaaclab_reindex]
    obs[idx:idx + joint_pos.shape[0]] = joint_pos
    idx += joint_pos.shape[0]

    # 8. joint_vel (29,) - relative joint velocities, reindexed to Isaac Lab order
    joint_vel = get_joint_vel_rel(sim_data)[mujoco_to_isaaclab_reindex]
    obs[idx:idx + joint_vel.shape[0]] = joint_vel
    idx += joint_vel.shape[0]

    # 9. actions (29,) - last actions
    last_action = get_last_action(last_actions)
    obs[idx:idx + last_action.shape[0]] = last_action
    idx += last_action.shape[0]

    return obs


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    VIEW_MOTION = False

    parser = argparse.ArgumentParser(description="Deploy MuJoCo simulation with motion tracking")
    parser.add_argument("--motion_path", type=str, default="", help="Path to motion file (.npz)")
    parser.add_argument(
        "--policy_path",
        type=str,
        default="",
        help="Path to policy file (.onnx)",
    )
    parser.add_argument(
        "--obs_config",
        type=str,
        default="ProjGravObs",
        choices=["ProjGravObs", "ProjGravAnchorObs"],
        help=(
            "Observation config to use. "
            "'ProjGravObs' (431-dim) for Tracking-Flat-G1-ProjGravObs-MNMLP-v0; "
            "'ProjGravAnchorObs' (440-dim) for Tracking-Flat-G1-ProjGravAnchorObs-NMMLP-v0 "
            "and Tracking-Flat-G1-ProjGravAnchorObs-NMMLP5L-v0."
        ),
    )
    parser.add_argument(
        "--anchor_body_name",
        type=str,
        default="pelvis",
        choices=["pelvis", "torso_link"],
        help="Anchor body used to build motion-anchor and robot-anchor observations.",
    )
    parser.add_argument(
        "--debug_obs",
        action="store_true",
        default=False,
        help="Print observation segments for debugging.",
    )
    parser.add_argument(
        "--debug_obs_steps",
        type=int,
        default=5,
        help="How many policy steps to print when --debug_obs is enabled.",
    )

    args = parser.parse_args()
    print(args)
    policy_path = args.policy_path
    motion_path = args.motion_path

    xml_path = "./source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1_act.xml"

    simulation_duration = 1000
    simulation_dt = 0.002
    control_decimation = 10

    num_actions = 29

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    counter = 0
    inner_counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    viewer = mujoco_viewer.MujocoViewer(m, d)
    viewer.cam.lookat[:] = np.array([0, 0, 0.7])
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -20  # 负值表示从上往下看viewer
    m.opt.timestep = simulation_dt

    # load policy
    session = ort.InferenceSession(policy_path)
    obs_name = session.get_inputs()[0].name

    motion_loader = MotionLoader(motion_path, anchor_body_name=args.anchor_body_name)
    T = motion_loader.T
    print("obs dim: ", session.get_inputs()[0])
    print("T: ", T)
    print()

    d.qpos[7:] = default_angles
    d.qpos[7:] = motion_loader.joint_pos[0][isaaclab_to_mujoco_reindex]
    # breakpoint()
    # d.qpos[:3] = np.array([0, 0, 0.76])
    d.qpos[:3] = motion_loader.body_pos[0][motion_loader.anchor_body_index]
    # d.qpos[2] = np.max((d.qpos[2], 0.80))
    # print(d.qpos[:3])
    # ori_yaw = motion_loader.body_ori[0][motion_loader.anchor_body_index]
    # ori_yaw[1:3] = 0
    # ori_yaw = ori_yaw / np.linalg.norm(ori_yaw)
    # d.qpos[3:7] = ori_yaw
    # breakpoint()
    d.qpos[3:7] = motion_loader.body_ori[0][motion_loader.anchor_body_index]
    mujoco.mj_step(m, d)

    # with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_alive and time.time() - start < simulation_duration:
        step_start = time.time()
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        if VIEW_MOTION:
            d.qpos[:3] = motion_loader.body_pos[inner_counter][0]
            d.qpos[3:7] = motion_loader.body_ori[inner_counter][0]
            d.qpos[7:] = motion_loader.joint_pos[inner_counter][isaaclab_to_mujoco_reindex]
            d.qvel[6:] = motion_loader.joint_vel[inner_counter][isaaclab_to_mujoco_reindex]
        else:
            d.ctrl[:] = tau

        mujoco.mj_step(m, d)
        counter += 1
        if counter % control_decimation == 0:
            # create observation
            # breakpoint()

            print("t: ", inner_counter, "/", T, end='\r')
            obs = compute_observation(d, motion_loader, inner_counter, action, args.obs_config)
            if args.debug_obs and inner_counter < args.debug_obs_steps:
                print_observation_debug(obs, args.obs_config, inner_counter)
            obs_tensor = np.array(obs, dtype=np.float32).reshape(1, -1)

            output = session.run(None, {obs_name: obs_tensor})
            action = output[0].squeeze()

            # transform action to target_dof_pos
            target_dof_pos = action[isaaclab_to_mujoco_reindex] * action_scale + default_angles

            inner_counter += 1

        if inner_counter >= motion_loader.T:
            # inner_counter -= 1
            inner_counter = 0
            # breakpoint()
            break

        # Update joint visualization
        update_joint_visualization(viewer, motion_loader, inner_counter)

        viewer.render()
