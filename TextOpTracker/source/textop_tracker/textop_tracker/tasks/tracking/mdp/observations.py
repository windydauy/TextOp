from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from textop_tracker.tasks.tracking.mdp import MotionCommand
from isaaclab.envs.mdp.observations import SceneEntityCfg
from isaaclab.envs.mdp.observations import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joint_vel_rel_no_ankle(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_rel = asset.data.joint_vel[:, :] - asset.data.default_joint_vel[:, :]
    # ankle_id  = asset_cfg.joint_ids
    print(joint_vel_rel[:, asset_cfg.joint_ids].norm(dim=-1).mean())
    # breakpoint()
    joint_vel_rel[:, asset_cfg.joint_ids] = 0
    return joint_vel_rel


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_pos_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor body position in world frame (single timestep)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_pos_w.view(env.num_envs, -1)


def robot_anchor_pos_w_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """N-step future reference anchor positions in world frame.

    Uses :attr:`MotionCommand.motion_anchor_pos` (same horizon and indexing as
    ``motion_anchor_pos_b_future``). Robot anchor and motion anchor share
    ``anchor_body_name`` in the command config.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.motion_anchor_pos.view(env.num_envs, -1)


def motion_body_pos_b(
    env: ManagerBasedEnv, command_name: str, body_names: list[str]
) -> torch.Tensor:
    """Reference motion body positions in robot anchor frame for the given body_names."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = [command.cfg.body_names.index(n) for n in body_names]
    n = len(body_indexes)

    robot_anchor_pos_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, n, -1)
    robot_anchor_quat_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, n, -1)

    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_exp,
        robot_anchor_quat_exp,
        command.body_pos_relative_w[:, body_indexes],
        command.body_quat_relative_w[:, body_indexes],
    )
    return pos_b.reshape(env.num_envs, -1)


def motion_body_ori_b(
    env: ManagerBasedEnv, command_name: str, body_names: list[str]
) -> torch.Tensor:
    """Reference motion body orientations (6D) in robot anchor frame for the given body_names."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = [command.cfg.body_names.index(n) for n in body_names]
    n = len(body_indexes)

    robot_anchor_pos_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, n, -1)
    robot_anchor_quat_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, n, -1)

    _, ori_b = subtract_frame_transforms(
        robot_anchor_pos_exp,
        robot_anchor_quat_exp,
        command.body_pos_relative_w[:, body_indexes],
        command.body_quat_relative_w[:, body_indexes],
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_body_pos_b_future(
    env: ManagerBasedEnv, command_name: str, body_names: list[str]
) -> torch.Tensor:
    """Future N-step reference body positions in robot anchor frame (same layout as ``motion_anchor_pos_b_future``)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = [command.cfg.body_names.index(n) for n in body_names]
    future_pos_w, future_quat_w = command.future_body_pos_quat_w(body_indexes)
    n_steps = future_pos_w.shape[1]
    n_bodies = future_pos_w.shape[2]

    robot_anchor_pos_exp = command.robot_anchor_pos_w[:, None, None, :].expand(
        -1, n_steps, n_bodies, -1
    )
    robot_anchor_quat_exp = command.robot_anchor_quat_w[:, None, None, :].expand(
        -1, n_steps, n_bodies, -1
    )
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_exp,
        robot_anchor_quat_exp,
        future_pos_w,
        future_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def motion_body_ori_b_future(
    env: ManagerBasedEnv, command_name: str, body_names: list[str]
) -> torch.Tensor:
    """Future N-step reference body orientations (6D) in robot anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = [command.cfg.body_names.index(n) for n in body_names]
    future_pos_w, future_quat_w = command.future_body_pos_quat_w(body_indexes)
    n_steps = future_pos_w.shape[1]
    n_bodies = future_pos_w.shape[2]

    robot_anchor_pos_exp = command.robot_anchor_pos_w[:, None, None, :].expand(
        -1, n_steps, n_bodies, -1
    )
    robot_anchor_quat_exp = command.robot_anchor_quat_w[:, None, None, :].expand(
        -1, n_steps, n_bodies, -1
    )
    _, ori_b = subtract_frame_transforms(
        robot_anchor_pos_exp,
        robot_anchor_quat_exp,
        future_pos_w,
        future_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_anchor_pos_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future N-step motion anchor position in body frame, flattened to vector."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Reshape future data: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    future_anchor_pos_w = command.motion_anchor_pos.view(env.num_envs, -1, 3)
    future_anchor_quat_w = command.motion_anchor_quat.view(env.num_envs, -1, 4)

    # Expand robot anchor for broadcasting: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    robot_anchor_pos_w_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, future_anchor_pos_w.shape[1], -1)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, future_anchor_quat_w.shape[1], -1)

    # Transform all future steps at once
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w_exp,
        robot_anchor_quat_w_exp,
        future_anchor_pos_w,
        future_anchor_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def motion_anchor_ori_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future N-step motion anchor orientation in body frame, flattened to vector."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Reshape future data: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    future_anchor_pos_w = command.motion_anchor_pos.view(env.num_envs, -1, 3)
    future_anchor_quat_w = command.motion_anchor_quat.view(env.num_envs, -1, 4)

    # Expand robot anchor for broadcasting: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    robot_anchor_pos_w_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, future_anchor_pos_w.shape[1], -1)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, future_anchor_quat_w.shape[1], -1)

    # Transform all future steps at once
    pos_b, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w_exp,
        robot_anchor_quat_w_exp,
        future_anchor_pos_w,
        future_anchor_quat_w,
    )

    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)
