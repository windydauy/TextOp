import glob
import re
import sys
import time
import mujoco, mujoco_viewer, mujoco.viewer
# `$ pip install mujoco-python-viewer` If `mujoco_viewer` not found.
import numpy as np
import torch
import argparse
from enum import Enum
from pathlib import Path


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate quaternion in wxyz convention."""
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Inverse quaternion in wxyz convention."""
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in wxyz convention."""
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in wxyz convention."""
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = torch.cross(xyz, vec, dim=-1) * 2.0
    return (vec + quat[:, 0:1] * t + torch.cross(xyz, t, dim=-1)).view(shape)


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / (quaternions * quaternions).sum(dim=-1)
    mat = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return mat.reshape(quaternions.shape[:-1] + (3, 3))


def subtract_frame_transforms(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor | None = None,
    q02: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute T12 = inv(T01) * T02 using wxyz quaternions."""
    q10 = quat_inv(q01)
    q12 = quat_mul(q10, q02) if q02 is not None else q10
    t12 = quat_apply(q10, t02 - t01) if t02 is not None else quat_apply(q10, -t01)
    return t12, q12


def load_onnxruntime():
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "onnxruntime is required to run the exported policy. Install it in this environment, "
            "for example: pip install onnxruntime or pip install onnxruntime-gpu"
        ) from exc
    return ort


NUM_ACTIONS = 29
COMMAND_DIM_PER_STEP = NUM_ACTIONS * 2
MOTION_AE_LATENT_DIM = 32
MOTION_TRANSFORMER_VAE_LATENT_DIM = 128
ANCHOR_POS_DIM_PER_STEP = 3
ANCHOR_ORI_DIM_PER_STEP = 6
PROJECTED_GRAVITY_DIM = 3
ROBOT_ANCHOR_POS_DIM = 3
ROBOT_ANCHOR_ORI_DIM = 6
BASE_LIN_VEL_DIM = 3
BASE_ANG_VEL_DIM = 3
JOINT_POS_DIM = NUM_ACTIONS
JOINT_VEL_DIM = NUM_ACTIONS
LAST_ACTION_DIM = NUM_ACTIONS

TASK_AUTO = "auto"
TASK_PROJ_GRAV_OBS = "Tracking-Flat-G1-ProjGravObs-MNMLP-v0"
TASK_PROJ_GRAV_ANCHOR_OBS = "Tracking-Flat-G1-ProjGravAnchorObs-NMMLP-v0"
TASK_PROJ_GRAV_ANCHOR_OBS_MOTION_AE = "Tracking-Flat-G1-ProjGravAnchorObs-MotionAE-NMMLP-v0"
TASK_PROJ_GRAV_ANCHOR_EE_OBS = "Tracking-Flat-G1-ProjGravAnchorEEObs-NMMLP-v0"
TASK_PROJ_GRAV_ANCHOR_OBS_TRANSFORMER_VAE = "Tracking-Flat-G1-ProjGravAnchorObs-TransformerVAE-NMMLP-v0"
SUPPORTED_TASKS = (
    TASK_PROJ_GRAV_OBS,
    TASK_PROJ_GRAV_ANCHOR_OBS,
    TASK_PROJ_GRAV_ANCHOR_OBS_MOTION_AE,
    TASK_PROJ_GRAV_ANCHOR_EE_OBS,
    TASK_PROJ_GRAV_ANCHOR_OBS_TRANSFORMER_VAE,
)

PROJ_GRAV_OBS_TERMS = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "projected_gravity",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)
PROJ_GRAV_ANCHOR_OBS_TERMS = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "robot_anchor_pos_w",
    "robot_anchor_ori_w",
    "projected_gravity",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)
PROJ_GRAV_ANCHOR_EE_OBS_TERMS = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "robot_anchor_pos_w",
    "robot_anchor_ori_w",
    "motion_ee_pos_b",
    "motion_ee_ori_b",
    "projected_gravity",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)
TASK_OBSERVATION_TERMS = {
    TASK_PROJ_GRAV_OBS: PROJ_GRAV_OBS_TERMS,
    TASK_PROJ_GRAV_ANCHOR_OBS: PROJ_GRAV_ANCHOR_OBS_TERMS,
    TASK_PROJ_GRAV_ANCHOR_OBS_MOTION_AE: PROJ_GRAV_ANCHOR_OBS_TERMS,
    TASK_PROJ_GRAV_ANCHOR_EE_OBS: PROJ_GRAV_ANCHOR_EE_OBS_TERMS,
    TASK_PROJ_GRAV_ANCHOR_OBS_TRANSFORMER_VAE: PROJ_GRAV_ANCHOR_OBS_TERMS,
}
FUTURE_TERM_DIMS = {
    "motion_anchor_pos_b": ANCHOR_POS_DIM_PER_STEP,
    "motion_anchor_ori_b": ANCHOR_ORI_DIM_PER_STEP,
    "motion_ee_pos_b": 4 * 3,
    "motion_ee_ori_b": 4 * 6,
}
FIXED_TERM_DIMS = {
    "projected_gravity": PROJECTED_GRAVITY_DIM,
    "robot_anchor_pos_w": ROBOT_ANCHOR_POS_DIM,
    "robot_anchor_ori_w": ROBOT_ANCHOR_ORI_DIM,
    "base_lin_vel": BASE_LIN_VEL_DIM,
    "base_ang_vel": BASE_ANG_VEL_DIM,
    "joint_pos": JOINT_POS_DIM,
    "joint_vel": JOINT_VEL_DIM,
    "actions": LAST_ACTION_DIM,
}

DEFAULT_MOTION_AE_PROJECT_ROOT = "/home/humanoid/yzh/TextOp/motion_ae"
DEFAULT_MOTION_AE_RUN_DIR = (
    "/home/humanoid/yzh/TextOp/motion_ae/outputs/motion_ae/"
    "2026-05-08_17-48-33_opti_clean_our_20"
)
DEFAULT_MOTION_AE_CONFIG_PATH = f"{DEFAULT_MOTION_AE_RUN_DIR}/params/config.yaml"
DEFAULT_MOTION_AE_STATS_PATH = f"{DEFAULT_MOTION_AE_RUN_DIR}/artifacts/stats.npz"
DEFAULT_MOTION_AE_ENCODER_ONNX_PATH = f"{DEFAULT_MOTION_AE_RUN_DIR}/artifacts/motion_ae_encoder_zdequant.onnx"

DEFAULT_MOTION_TRANSFORMER_VAE_PROJECT_ROOT = "/home/humanoid/yzh/TextOp/motion_ae/transformer_vae"
DEFAULT_MOTION_TRANSFORMER_VAE_RUN_DIR = (
    "/home/humanoid/yzh/TextOp/motion_ae/outputs/transformer_vae/"
    "2026-05-11_00-19-00_optitrack_npz_trip_filtered"
)
DEFAULT_MOTION_TRANSFORMER_VAE_CONFIG_PATH = f"{DEFAULT_MOTION_TRANSFORMER_VAE_RUN_DIR}/params/config.yaml"
DEFAULT_MOTION_TRANSFORMER_VAE_STATS_PATH = f"{DEFAULT_MOTION_TRANSFORMER_VAE_RUN_DIR}/artifacts/stats.npz"
DEFAULT_MOTION_TRANSFORMER_VAE_ENCODER_ONNX_PATH = (
    f"{DEFAULT_MOTION_TRANSFORMER_VAE_RUN_DIR}/artifacts/motion_transformer_vae_encoder_z_c.onnx"
)


# 定义枚举类型
class AnchorBody(Enum):
    PELVIS = 0
    TORSO_LINK = 9


anchor_body = AnchorBody.PELVIS

ANCHOR_BODY_INDEX_BY_NAME = {
    "pelvis": AnchorBody.PELVIS.value,
    "torso_link": AnchorBody.TORSO_LINK.value,
}

DEFAULT_BODY_NAMES = [
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

DEFAULT_EE_BODY_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

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


def _csv_metadata_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _decode_body_names(raw_names):
    if raw_names is None:
        return []
    return [
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in raw_names
    ]


def get_policy_metadata(session):
    try:
        return dict(session.get_modelmeta().custom_metadata_map)
    except Exception:
        return {}


def get_policy_input_dim(session):
    shape = session.get_inputs()[0].shape
    if len(shape) < 2:
        return None
    dim = shape[1]
    return dim if isinstance(dim, int) else None


def _uses_projected_gravity(obs_config):
    return "ProjGrav" in obs_config


def _uses_robot_anchor_obs(obs_config):
    return "AnchorObs" in obs_config


def _candidate_tasks_from_observation_names(observation_names):
    names = tuple(observation_names or ())
    if not names:
        return []
    candidates = [task for task, terms in TASK_OBSERVATION_TERMS.items() if names == terms]
    if candidates:
        return candidates
    if "robot_anchor_pos_w" in names or "robot_anchor_ori_w" in names:
        candidates = [
            TASK_PROJ_GRAV_ANCHOR_OBS,
            TASK_PROJ_GRAV_ANCHOR_OBS_MOTION_AE,
            TASK_PROJ_GRAV_ANCHOR_EE_OBS,
            TASK_PROJ_GRAV_ANCHOR_OBS_TRANSFORMER_VAE,
        ]
        if "motion_ee_pos_b" in names or "motion_ee_ori_b" in names:
            return [TASK_PROJ_GRAV_ANCHOR_EE_OBS]
        return candidates
    if "projected_gravity" in names:
        return [TASK_PROJ_GRAV_OBS]
    return []


def infer_task_from_observation_names(observation_names, input_dim=None, future_steps=None):
    candidates = _candidate_tasks_from_observation_names(observation_names)
    if not candidates:
        return None
    if input_dim is not None:
        matches = []
        for candidate in candidates:
            try:
                if future_steps is None:
                    infer_future_steps_from_input_dim(input_dim, task=candidate)
                    matches.append(candidate)
                elif compute_observation_dim(future_steps, task=candidate) == input_dim:
                    matches.append(candidate)
            except ValueError:
                continue
        if len(matches) == 1:
            return matches[0]
    return candidates[0]


def task_from_obs_config(obs_config):
    if obs_config in TASK_OBSERVATION_TERMS:
        return obs_config
    if "EEObs" in obs_config:
        return TASK_PROJ_GRAV_ANCHOR_EE_OBS
    if _uses_robot_anchor_obs(obs_config):
        return TASK_PROJ_GRAV_ANCHOR_OBS
    if _uses_projected_gravity(obs_config):
        return TASK_PROJ_GRAV_OBS
    raise ValueError(f"Unsupported obs_config={obs_config}. Please pass --task explicitly.")


def resolve_task(task=TASK_AUTO, policy_metadata=None, obs_config="ProjGravObs", policy_input_dim=None, future_steps=None):
    if task != TASK_AUTO:
        if task not in TASK_OBSERVATION_TERMS:
            choices = ", ".join(SUPPORTED_TASKS)
            raise ValueError(f"Unsupported task={task}. Supported tasks: {choices}")
        return task

    metadata_obs_names = []
    if policy_metadata:
        metadata_obs_names = _csv_metadata_list(policy_metadata.get("observation_names"))
    inferred_task = infer_task_from_observation_names(
        metadata_obs_names,
        input_dim=policy_input_dim,
        future_steps=future_steps,
    )
    if inferred_task is not None:
        return inferred_task
    return task_from_obs_config(obs_config)


def get_observation_terms(task=None, obs_config="ProjGravObs", observation_terms=None):
    if observation_terms is not None:
        return tuple(observation_terms)
    resolved_task = task or task_from_obs_config(obs_config)
    try:
        return TASK_OBSERVATION_TERMS[resolved_task]
    except KeyError as exc:
        choices = ", ".join(SUPPORTED_TASKS)
        raise ValueError(f"Unsupported task={resolved_task}. Supported tasks: {choices}") from exc


def _term_dims(observation_terms, task=None):
    fixed_dim = 0
    per_step_dim = 0
    for term in observation_terms:
        if term == "command":
            if task in (TASK_PROJ_GRAV_ANCHOR_OBS_MOTION_AE, TASK_PROJ_GRAV_ANCHOR_EE_OBS):
                fixed_dim += MOTION_AE_LATENT_DIM
            elif task == TASK_PROJ_GRAV_ANCHOR_OBS_TRANSFORMER_VAE:
                fixed_dim += MOTION_TRANSFORMER_VAE_LATENT_DIM
            else:
                per_step_dim += COMMAND_DIM_PER_STEP
        elif term in FUTURE_TERM_DIMS:
            per_step_dim += FUTURE_TERM_DIMS[term]
        elif term in FIXED_TERM_DIMS:
            fixed_dim += FIXED_TERM_DIMS[term]
        else:
            raise ValueError(f"Unsupported observation term={term}")
    return fixed_dim, per_step_dim


def compute_observation_dim(future_steps, obs_config="ProjGravObs", task=None, observation_terms=None):
    terms = get_observation_terms(task=task, obs_config=obs_config, observation_terms=observation_terms)
    fixed_dim, per_step_dim = _term_dims(terms, task=task)
    return future_steps * per_step_dim + fixed_dim


def infer_future_steps_from_input_dim(input_dim, obs_config="ProjGravObs", task=None, observation_terms=None):
    if input_dim is None:
        return None
    terms = get_observation_terms(task=task, obs_config=obs_config, observation_terms=observation_terms)
    fixed_dim, per_step_dim = _term_dims(terms, task=task)
    remainder = input_dim - fixed_dim
    if remainder <= 0 or remainder % per_step_dim != 0:
        raise ValueError(
            f"Cannot infer future_steps from ONNX input dim {input_dim} with task={task}, obs_config={obs_config}. "
            f"Fixed dim={fixed_dim}, per-step dim={per_step_dim}."
        )
    return remainder // per_step_dim


def resolve_motion_file(motion_path):
    path = Path(motion_path)
    candidates = []

    if path.is_file():
        return str(path)
    if path.is_dir():
        candidates.append(path / "motion.npz")
    if any(ch in motion_path for ch in "*?[]"):
        candidates.extend(Path(p) for p in glob.glob(motion_path))
        candidates.extend(Path(p) for p in glob.glob(str(path / "motion.npz")))

    for candidate in sorted(candidates):
        if candidate.is_dir():
            candidate = candidate / "motion.npz"
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(f"No motion.npz found from motion_path={motion_path}")


def get_motion_anchor_body_index(anchor_body_name):
    try:
        return ANCHOR_BODY_INDEX_BY_NAME[anchor_body_name]
    except KeyError as exc:
        choices = ", ".join(sorted(ANCHOR_BODY_INDEX_BY_NAME))
        raise ValueError(f"Unsupported anchor_body_name={anchor_body_name}. Known anchors: {choices}") from exc


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


class MotionAEOnnxLatentAdapter:
    """Encode one motion.npz into MotionAE z_dequant latents using an encoder ONNX."""

    def __init__(
        self,
        *,
        project_root,
        config_path,
        stats_path,
        encoder_onnx_path,
        batch_size=4096,
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.stats_path = Path(stats_path).expanduser().resolve()
        self.encoder_onnx_path = Path(encoder_onnx_path).expanduser().resolve()
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"motion_ae_batch_size must be positive, got {batch_size}")
        self._validate_paths()
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        from motion_ae.config import load_config
        from motion_ae.utils.normalization import FeatureNormalizer

        self.cfg = load_config(str(self.config_path))
        self.normalizer = FeatureNormalizer.load(str(self.stats_path), eps=self.cfg.normalization.eps)
        self.window_size = int(self.cfg.window_size)
        self.feature_dim = int(self.normalizer.mean.shape[0])
        self.latent_dim = int(self.cfg.model.latent_dim)

        ort = load_onnxruntime()
        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(self.encoder_onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.z_dequant_output_name = "z_dequant" if "z_dequant" in self.output_names else self.output_names[0]

    def _validate_paths(self):
        for label, path in {
            "MotionAE project root": self.project_root,
            "MotionAE config": self.config_path,
            "MotionAE stats": self.stats_path,
            "MotionAE encoder ONNX": self.encoder_onnx_path,
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

    def encode_npz_data(self, npz_data):
        from motion_ae.feature_builder import build_features

        features, slices = build_features(
            npz_data,
            self.cfg.npz_keys,
            self.cfg.pelvis,
            debug=False,
        )
        if slices.total_dim != self.feature_dim:
            raise ValueError(
                f"MotionAE feature dim mismatch: encoder expects {self.feature_dim}, "
                f"motion produced {slices.total_dim}"
            )
        if features.shape[0] == 0:
            raise ValueError("Cannot encode empty motion")

        normalized = self.normalizer.normalize_np(features).astype(np.float32, copy=False)
        windows = self._future_windows(normalized)
        chunks = []
        for start in range(0, windows.shape[0], self.batch_size):
            batch = windows[start:start + self.batch_size]
            latent = self.session.run([self.z_dequant_output_name], {self.input_name: batch})[0]
            chunks.append(np.asarray(latent, dtype=np.float32))
        latents = np.concatenate(chunks, axis=0)
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(f"MotionAE latent shape mismatch: expected [T, {self.latent_dim}], got {latents.shape}")
        if not np.isfinite(latents).all():
            raise ValueError("MotionAE latents contain non-finite values")
        return latents

    def _future_windows(self, features):
        frame_count = int(features.shape[0])
        frame_ids = np.arange(frame_count, dtype=np.int64)[:, None]
        offsets = np.arange(self.window_size, dtype=np.int64)[None, :]
        indices = np.clip(frame_ids + offsets, 0, frame_count - 1)
        return features[indices].astype(np.float32, copy=False)


class MotionTransformerVAEOnnxLatentAdapter:
    """Encode one motion.npz into Transformer VAE z_c latents using an encoder ONNX."""

    def __init__(
        self,
        *,
        project_root,
        config_path,
        stats_path,
        encoder_onnx_path,
        batch_size=4096,
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.stats_path = Path(stats_path).expanduser().resolve()
        self.encoder_onnx_path = Path(encoder_onnx_path).expanduser().resolve()
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"motion_transformer_vae_batch_size must be positive, got {batch_size}")
        self._validate_paths()
        for import_root in (self.project_root.parent, self.project_root):
            import_root_str = str(import_root)
            if import_root_str not in sys.path:
                sys.path.insert(0, import_root_str)

        from motion_ae.utils.normalization import FeatureNormalizer
        from transformer_vae.config import load_config

        self.cfg = load_config(str(self.config_path))
        self.normalizer = FeatureNormalizer.load(str(self.stats_path), eps=self.cfg.normalization.eps)
        self.window_size = int(self.cfg.window_size)
        self.feature_dim = int(self.normalizer.mean.shape[0])
        self.latent_size = int(self.cfg.model.latent_dim[0])
        self.latent_dim = int(self.cfg.model.latent_dim[-1])
        self.flat_latent_dim = self.latent_size * self.latent_dim

        ort = load_onnxruntime()
        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(self.encoder_onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.z_c_output_name = "z_c" if "z_c" in self.output_names else self.output_names[0]

    def _validate_paths(self):
        for label, path in {
            "MotionTransformerVAE project root": self.project_root,
            "MotionTransformerVAE config": self.config_path,
            "MotionTransformerVAE stats": self.stats_path,
            "MotionTransformerVAE encoder ONNX": self.encoder_onnx_path,
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

    def encode_npz_data(self, npz_data):
        from motion_ae.feature_builder import build_features

        features, slices = build_features(
            npz_data,
            self.cfg.npz_keys,
            self.cfg.pelvis,
            debug=False,
        )
        if slices.total_dim != self.feature_dim:
            raise ValueError(
                f"MotionTransformerVAE feature dim mismatch: encoder expects {self.feature_dim}, "
                f"motion produced {slices.total_dim}"
            )
        if features.shape[0] == 0:
            raise ValueError("Cannot encode empty motion")

        normalized = self.normalizer.normalize_np(features).astype(np.float32, copy=False)
        windows = self._future_windows(normalized)
        chunks = []
        for start in range(0, windows.shape[0], self.batch_size):
            batch = windows[start:start + self.batch_size]
            latent = self.session.run([self.z_c_output_name], {self.input_name: batch})[0]
            chunks.append(np.asarray(latent, dtype=np.float32))
        latents = np.concatenate(chunks, axis=0)
        if latents.ndim != 2 or latents.shape[1] != self.flat_latent_dim:
            raise ValueError(
                "MotionTransformerVAE latent shape mismatch: "
                f"expected [T, {self.flat_latent_dim}], got {latents.shape}"
            )
        if not np.isfinite(latents).all():
            raise ValueError("MotionTransformerVAE latents contain non-finite values")
        return latents

    def _future_windows(self, features):
        frame_count = int(features.shape[0])
        frame_ids = np.arange(frame_count, dtype=np.int64)[:, None]
        offsets = np.arange(self.window_size, dtype=np.int64)[None, :]
        indices = np.clip(frame_ids + offsets, 0, frame_count - 1)
        return features[indices].astype(np.float32, copy=False)


# ====== MotionLoader 类（参考 isaaclab） ======
class MotionLoader:
    def __init__(
        self,
        motion_file,
        future_steps,
        anchor_body_name,
        body_names=None,
        motion_ae_adapter=None,
        motion_transformer_vae_adapter=None,
    ):
        motion_file = resolve_motion_file(motion_file)
        print(motion_file)
        data = np.load(motion_file)
        self.joint_pos = data["joint_pos"]  # [T, 29]
        self.joint_vel = data["joint_vel"]  # [T, 29]
        self.body_pos = data["body_pos_w"]  # [T, N, 3]
        self.body_ori = data["body_quat_w"]  # [T, N, 4]
        # self.body_ang_vel_w = data["body_ang_vel_w"]
        self.fps = data["fps"]
        self.T = self.joint_pos.shape[0]

        # self.body_pos[..., 2] += 0.015

        # G1 body names from the config
        self.body_names = body_names or DEFAULT_BODY_NAMES
        self.anchor_body_name = anchor_body_name
        motion_body_names = _decode_body_names(data["body_names"]) if "body_names" in data.files else []
        if motion_body_names:
            self.motion_body_index_by_name = {name: i for i, name in enumerate(motion_body_names)}
        else:
            self.motion_body_index_by_name = {name: i for i, name in enumerate(DEFAULT_BODY_NAMES)}
        self.anchor_body_index = self.motion_body_index(anchor_body_name)
        if self.anchor_body_index >= self.body_pos.shape[1]:
            raise ValueError(
                f"Motion body count is {self.body_pos.shape[1]}, but anchor {anchor_body_name} "
                f"needs index {self.anchor_body_index}."
            )

        # Future steps configuration
        self.future_steps = future_steps
        self.motion_ae_latents = None
        if motion_ae_adapter is not None and motion_transformer_vae_adapter is not None:
            raise ValueError("Only one latent adapter can be enabled for deploy.")
        if motion_ae_adapter is not None:
            if motion_ae_adapter.window_size != self.future_steps:
                raise ValueError(
                    "MotionAE encoder window_size must match deploy future_steps: "
                    f"{motion_ae_adapter.window_size} vs {self.future_steps}"
                )
            self.motion_ae_latents = motion_ae_adapter.encode_npz_data(data)
            if self.motion_ae_latents.shape[0] != self.T:
                raise ValueError(
                    f"MotionAE latent length mismatch: {self.motion_ae_latents.shape[0]} vs motion frames {self.T}"
                )
        if motion_transformer_vae_adapter is not None:
            if motion_transformer_vae_adapter.window_size != self.future_steps:
                raise ValueError(
                    "MotionTransformerVAE encoder window_size must match deploy future_steps: "
                    f"{motion_transformer_vae_adapter.window_size} vs {self.future_steps}"
                )
            self.motion_ae_latents = motion_transformer_vae_adapter.encode_npz_data(data)
            if self.motion_ae_latents.shape[0] != self.T:
                raise ValueError(
                    "MotionTransformerVAE latent length mismatch: "
                    f"{self.motion_ae_latents.shape[0]} vs motion frames {self.T}"
                )

    def motion_body_index(self, body_name):
        if body_name in self.motion_body_index_by_name:
            return self.motion_body_index_by_name[body_name]
        return get_motion_anchor_body_index(body_name)


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


# ====== observation 计算函数 ======
def get_command(motion_loader, t):
    """Get command (joint_pos + joint_vel) for future steps - matching Isaac Lab order"""
    if motion_loader.motion_ae_latents is not None:
        if t < 0:
            return np.zeros(motion_loader.motion_ae_latents.shape[1], dtype=np.float32)
        step_idx = min(t, motion_loader.T - 1)
        return motion_loader.motion_ae_latents[step_idx].astype(np.float32, copy=False)

    if t < 0:
        return np.zeros(motion_loader.future_steps * COMMAND_DIM_PER_STEP, dtype=np.float32)

    # Get future joint positions and velocities - batch process like Isaac Lab
    future_joint_pos = []
    future_joint_vel = []

    for i in range(motion_loader.future_steps):
        step_idx = min(t + i, motion_loader.T - 1)
        future_joint_pos.append(motion_loader.joint_pos[step_idx])
        future_joint_vel.append(motion_loader.joint_vel[step_idx])

    # Stack to create [future_steps, 29] then flatten to [future_steps * 29]
    joint_pos_future = np.stack(future_joint_pos, axis=0).flatten()
    joint_vel_future = np.stack(future_joint_vel, axis=0).flatten()
    # breakpoint()

    cmd = np.concatenate([joint_pos_future, joint_vel_future], axis=0)
    # breakpoint()
    return cmd  # [290]


def motion_anchor_pos_b_future(sim_data, motion_loader, t):
    """Future N-step motion anchor position in body frame - matching Isaac Lab order"""
    if t < 0:
        return np.zeros(motion_loader.future_steps * ANCHOR_POS_DIM_PER_STEP, dtype=np.float32)

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
    pos_b, _ = subtract_frame_transforms(
        torch.from_numpy(robot_anchor_pos_w_exp).float(),
        torch.from_numpy(robot_anchor_quat_w_exp).float(),
        torch.from_numpy(future_anchor_pos_w).float(),
        torch.from_numpy(future_anchor_quat_w).float(),
    )

    pos_b_flat = np.array(pos_b, dtype=np.float32).flatten()  # [15]
    return pos_b_flat  # [15]


def motion_anchor_ori_b_future(sim_data, motion_loader, t):
    """Future N-step motion anchor orientation in body frame - matching Isaac Lab order"""
    if t < 0:
        return np.zeros(motion_loader.future_steps * ANCHOR_ORI_DIM_PER_STEP, dtype=np.float32)

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
    pos_b, ori_b = subtract_frame_transforms(
        torch.from_numpy(robot_anchor_pos_w_exp).float(),
        torch.from_numpy(robot_anchor_quat_w_exp).float(),
        torch.from_numpy(future_anchor_pos_w).float(),
        torch.from_numpy(future_anchor_quat_w).float(),
    )

    # Convert to rotation matrix and take first 2 rows
    mat = matrix_from_quat(ori_b)
    mat_flat = mat[..., :2].reshape(-1)
    # breakpoint()
    ori_b_flat = np.array(mat_flat, dtype=np.float32).flatten()  # [30]
    # breakpoint()
    return ori_b_flat  # [30]


def motion_body_pos_b_future(sim_data, motion_loader, t, body_names):
    """Future N-step reference body positions in robot anchor frame."""
    if t < 0:
        return np.zeros(motion_loader.future_steps * len(body_names) * 3, dtype=np.float32)

    robot_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy().reshape(1, 1, 3)
    robot_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 1, 4)
    body_indexes = [motion_loader.motion_body_index(name) for name in body_names]

    future_positions = []
    future_orientations = []
    for i in range(motion_loader.future_steps):
        step_idx = min(t + i, motion_loader.T - 1)
        future_positions.append(motion_loader.body_pos[step_idx, body_indexes])
        future_orientations.append(motion_loader.body_ori[step_idx, body_indexes])

    future_body_pos_w = np.stack(future_positions, axis=0)
    future_body_quat_w = np.stack(future_orientations, axis=0)
    robot_anchor_pos_w_exp = np.repeat(robot_pos, motion_loader.future_steps, axis=0)
    robot_anchor_pos_w_exp = np.repeat(robot_anchor_pos_w_exp, len(body_names), axis=1)
    robot_anchor_quat_w_exp = np.repeat(robot_quat, motion_loader.future_steps, axis=0)
    robot_anchor_quat_w_exp = np.repeat(robot_anchor_quat_w_exp, len(body_names), axis=1)

    pos_b, _ = subtract_frame_transforms(
        torch.from_numpy(robot_anchor_pos_w_exp).float(),
        torch.from_numpy(robot_anchor_quat_w_exp).float(),
        torch.from_numpy(future_body_pos_w).float(),
        torch.from_numpy(future_body_quat_w).float(),
    )
    return np.array(pos_b, dtype=np.float32).reshape(-1)


def motion_body_ori_b_future(sim_data, motion_loader, t, body_names):
    """Future N-step reference body orientations in robot anchor frame, as 6D rotation."""
    if t < 0:
        return np.zeros(motion_loader.future_steps * len(body_names) * 6, dtype=np.float32)

    robot_pos = sim_data.body(motion_loader.anchor_body_name).xpos.copy().reshape(1, 1, 3)
    robot_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 1, 4)
    body_indexes = [motion_loader.motion_body_index(name) for name in body_names]

    future_positions = []
    future_orientations = []
    for i in range(motion_loader.future_steps):
        step_idx = min(t + i, motion_loader.T - 1)
        future_positions.append(motion_loader.body_pos[step_idx, body_indexes])
        future_orientations.append(motion_loader.body_ori[step_idx, body_indexes])

    future_body_pos_w = np.stack(future_positions, axis=0)
    future_body_quat_w = np.stack(future_orientations, axis=0)
    robot_anchor_pos_w_exp = np.repeat(robot_pos, motion_loader.future_steps, axis=0)
    robot_anchor_pos_w_exp = np.repeat(robot_anchor_pos_w_exp, len(body_names), axis=1)
    robot_anchor_quat_w_exp = np.repeat(robot_quat, motion_loader.future_steps, axis=0)
    robot_anchor_quat_w_exp = np.repeat(robot_anchor_quat_w_exp, len(body_names), axis=1)

    _, ori_b = subtract_frame_transforms(
        torch.from_numpy(robot_anchor_pos_w_exp).float(),
        torch.from_numpy(robot_anchor_quat_w_exp).float(),
        torch.from_numpy(future_body_pos_w).float(),
        torch.from_numpy(future_body_quat_w).float(),
    )
    mat = matrix_from_quat(ori_b)
    return np.array(mat[..., :2].reshape(-1), dtype=np.float32)


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

    pos_b, _ = subtract_frame_transforms(
        torch.from_numpy(robot_anchor_pos_expanded).float(),
        torch.from_numpy(robot_anchor_quat_expanded).float(),
        torch.from_numpy(robot_body_pos).float(),
        torch.from_numpy(robot_body_quat).float(),
    )

    return np.array(pos_b, dtype=np.float32).flatten()  # [42]


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

    _, ori_b = subtract_frame_transforms(
        torch.from_numpy(robot_anchor_pos_expanded).float(),
        torch.from_numpy(robot_anchor_quat_expanded).float(),
        torch.from_numpy(robot_body_pos).float(),
        torch.from_numpy(robot_body_quat).float(),
    )

    # Convert to rotation matrix and take first 2 rows
    mat = matrix_from_quat(ori_b)
    mat_flat = mat[..., :2].reshape(-1)

    return np.array(mat_flat, dtype=np.float32).flatten()  # [84]


def robot_anchor_pos_w(sim_data, motion_loader):
    """Robot anchor position in world frame."""
    return sim_data.body(motion_loader.anchor_body_name).xpos.copy().astype(np.float32)


def robot_anchor_ori_w(sim_data, motion_loader):
    """Robot anchor orientation in world frame, represented as first two rows of the rotation matrix."""
    robot_quat = sim_data.body(motion_loader.anchor_body_name).xquat.copy().reshape(1, 4)
    mat = matrix_from_quat(torch.from_numpy(robot_quat).float())
    return np.array(mat[..., :2].reshape(-1), dtype=np.float32)


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


#


def compute_observation_term(term, sim_data, motion_loader, t, last_actions):
    """Compute one policy observation term in Isaac Lab policy order."""
    if term == "command":
        return get_command(motion_loader, t)
    if term == "motion_anchor_pos_b":
        return motion_anchor_pos_b_future(sim_data, motion_loader, t)
    if term == "motion_anchor_ori_b":
        return motion_anchor_ori_b_future(sim_data, motion_loader, t)
    if term == "robot_anchor_pos_w":
        return robot_anchor_pos_w(sim_data, motion_loader)
    if term == "robot_anchor_ori_w":
        return robot_anchor_ori_w(sim_data, motion_loader)
    if term == "motion_ee_pos_b":
        return motion_body_pos_b_future(sim_data, motion_loader, t, DEFAULT_EE_BODY_NAMES)
    if term == "motion_ee_ori_b":
        return motion_body_ori_b_future(sim_data, motion_loader, t, DEFAULT_EE_BODY_NAMES)
    if term == "projected_gravity":
        return get_projected_gravity(sim_data)
    if term == "base_lin_vel":
        return get_base_lin_vel(sim_data)
    if term == "base_ang_vel":
        return get_base_ang_vel(sim_data)
    if term == "joint_pos":
        return get_joint_pos_rel(sim_data)[mujoco_to_isaaclab_reindex]
    if term == "joint_vel":
        return get_joint_vel_rel(sim_data)[mujoco_to_isaaclab_reindex]
    if term == "actions":
        return get_last_action(last_actions)
    raise ValueError(f"Unsupported observation term={term}")


def compute_observation(
    sim_data,
    motion_loader,
    t,
    last_actions,
    obs_config="ProjGravObs",
    task=None,
    observation_terms=None,
):
    """Compute policy observation in the same order as the Isaac Lab policy group."""
    terms = get_observation_terms(task=task, obs_config=obs_config, observation_terms=observation_terms)
    obs_dim = compute_observation_dim(motion_loader.future_steps, obs_config, task=task, observation_terms=terms)

    if t < 0:
        return np.zeros(obs_dim, dtype=np.float32)

    pieces = [
        np.asarray(compute_observation_term(term, sim_data, motion_loader, t, last_actions), dtype=np.float32).reshape(-1)
        for term in terms
    ]
    obs = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)

    if obs.shape[0] != obs_dim:
        raise RuntimeError(f"Observation assembly produced dim {obs.shape[0]}, expected {obs_dim}.")

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
    parser.add_argument("--future_steps", type=int, default=None, help="Future steps used by the exported policy.")
    parser.add_argument("--obs_config", type=str, default="ProjGravObs", help="Observation config used during training.")
    parser.add_argument(
        "--task",
        type=str,
        default=TASK_AUTO,
        choices=(TASK_AUTO, *SUPPORTED_TASKS),
        help="Training task that defines the deploy observation layout. Use auto to infer from ONNX metadata.",
    )
    parser.add_argument("--anchor_body_name", type=str, default=None, help="Anchor body name used by the motion command.")
    parser.add_argument("--motion_ae_encoder_onnx_path", type=str, default=DEFAULT_MOTION_AE_ENCODER_ONNX_PATH)
    parser.add_argument("--motion_ae_project_root", type=str, default=DEFAULT_MOTION_AE_PROJECT_ROOT)
    parser.add_argument("--motion_ae_config_path", type=str, default=DEFAULT_MOTION_AE_CONFIG_PATH)
    parser.add_argument("--motion_ae_stats_path", type=str, default=DEFAULT_MOTION_AE_STATS_PATH)
    parser.add_argument("--motion_ae_batch_size", type=int, default=4096)
    parser.add_argument(
        "--motion_transformer_vae_encoder_onnx_path",
        type=str,
        default=DEFAULT_MOTION_TRANSFORMER_VAE_ENCODER_ONNX_PATH,
    )
    parser.add_argument(
        "--motion_transformer_vae_project_root",
        type=str,
        default=DEFAULT_MOTION_TRANSFORMER_VAE_PROJECT_ROOT,
    )
    parser.add_argument(
        "--motion_transformer_vae_config_path",
        type=str,
        default=DEFAULT_MOTION_TRANSFORMER_VAE_CONFIG_PATH,
    )
    parser.add_argument(
        "--motion_transformer_vae_stats_path",
        type=str,
        default=DEFAULT_MOTION_TRANSFORMER_VAE_STATS_PATH,
    )
    parser.add_argument("--motion_transformer_vae_batch_size", type=int, default=4096)

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
    ort = load_onnxruntime()
    session = ort.InferenceSession(policy_path)
    obs_name = session.get_inputs()[0].name
    policy_metadata = get_policy_metadata(session)
    policy_input_dim = get_policy_input_dim(session)
    task = resolve_task(
        args.task,
        policy_metadata=policy_metadata,
        obs_config=args.obs_config,
        policy_input_dim=policy_input_dim,
        future_steps=args.future_steps,
    )
    observation_terms = get_observation_terms(task=task)

    anchor_body_name = args.anchor_body_name or policy_metadata.get("anchor_body_name", "pelvis")
    body_names = _csv_metadata_list(policy_metadata.get("body_names")) or DEFAULT_BODY_NAMES
    future_steps = args.future_steps
    if future_steps is None:
        future_steps = infer_future_steps_from_input_dim(
            policy_input_dim,
            obs_config=args.obs_config,
            task=task,
            observation_terms=observation_terms,
        )
    if future_steps is None:
        raise ValueError("Unable to infer future_steps from ONNX input shape. Please pass --future_steps explicitly.")
    expected_obs_dim = compute_observation_dim(
        future_steps,
        obs_config=args.obs_config,
        task=task,
        observation_terms=observation_terms,
    )
    if policy_input_dim is not None and policy_input_dim != expected_obs_dim:
        raise ValueError(
            f"Deploy observation dim {expected_obs_dim} does not match ONNX input dim {policy_input_dim}. "
            f"task={task}, future_steps={future_steps}, obs_config={args.obs_config}. "
            f"Check --task and --future_steps."
        )

    motion_ae_adapter = None
    motion_transformer_vae_adapter = None
    if task in (TASK_PROJ_GRAV_ANCHOR_OBS_MOTION_AE, TASK_PROJ_GRAV_ANCHOR_EE_OBS):
        motion_ae_adapter = MotionAEOnnxLatentAdapter(
            project_root=args.motion_ae_project_root,
            config_path=args.motion_ae_config_path,
            stats_path=args.motion_ae_stats_path,
            encoder_onnx_path=args.motion_ae_encoder_onnx_path,
            batch_size=args.motion_ae_batch_size,
        )
    elif task == TASK_PROJ_GRAV_ANCHOR_OBS_TRANSFORMER_VAE:
        motion_transformer_vae_adapter = MotionTransformerVAEOnnxLatentAdapter(
            project_root=args.motion_transformer_vae_project_root,
            config_path=args.motion_transformer_vae_config_path,
            stats_path=args.motion_transformer_vae_stats_path,
            encoder_onnx_path=args.motion_transformer_vae_encoder_onnx_path,
            batch_size=args.motion_transformer_vae_batch_size,
        )

    motion_loader = MotionLoader(
        motion_path,
        future_steps=future_steps,
        anchor_body_name=anchor_body_name,
        body_names=body_names,
        motion_ae_adapter=motion_ae_adapter,
        motion_transformer_vae_adapter=motion_transformer_vae_adapter,
    )
    T = motion_loader.T
    print("obs dim: ", session.get_inputs()[0])
    print("deploy task:", task)
    print("deploy obs_config:", args.obs_config)
    print("deploy observation terms:", ",".join(observation_terms))
    print("deploy future_steps:", future_steps)
    print("deploy anchor_body_name:", anchor_body_name)
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
            obs = compute_observation(
                d,
                motion_loader,
                inner_counter,
                action,
                obs_config=args.obs_config,
                task=task,
                observation_terms=observation_terms,
            )
            obs_tensor = np.array(obs, dtype=np.float32).reshape(1, -1)
            if policy_input_dim is not None and obs_tensor.shape[1] != policy_input_dim:
                raise RuntimeError(
                    f"Computed obs dim {obs_tensor.shape[1]} does not match ONNX input dim {policy_input_dim}."
                )

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
