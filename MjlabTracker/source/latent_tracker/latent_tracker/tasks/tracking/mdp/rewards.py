from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from typing import Callable

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    yaw_quat,
)
from latent_tracker.tasks.tracking.mdp.observations import (
    motion_anchor_ori_b_future as obs_motion_anchor_ori_b_future,
    motion_anchor_pos_b_future as obs_motion_anchor_pos_b_future,
    motion_body_ori_b_future as obs_motion_body_ori_b_future,
    motion_body_pos_b_future as obs_motion_body_pos_b_future,
    robot_anchor_ori_w as obs_robot_anchor_ori_w,
    robot_anchor_pos_w as obs_robot_anchor_pos_w,
)
from latent_tracker.tasks.tracking.mdp import MotionCommand
if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def contact_forces(env: ManagerBasedRlEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Count contact slots whose force exceeds ``threshold`` using mjlab sensors."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        hit = (force_mag > threshold).any(dim=-1)
        return hit[:, sensor_cfg.body_ids].sum(dim=-1).float()
    assert data.force is not None
    hit = torch.norm(data.force, dim=-1) > threshold
    return hit[:, sensor_cfg.body_ids].sum(dim=-1).float()


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _matvec(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return torch.matmul(mat, vec.unsqueeze(-1)).squeeze(-1)


def _rotation_vector_from_matrix(rot: torch.Tensor) -> torch.Tensor:
    trace = rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    skew_vec = torch.stack(
        (
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
        ),
        dim=-1,
    )
    sin_angle = torch.sin(angle)
    scale = torch.where(
        sin_angle.abs() > 1e-6,
        angle / (2.0 * sin_angle),
        torch.full_like(angle, 0.5),
    )
    return skew_vec * scale.unsqueeze(-1)


def _rotation_error_squared(ref_rot: torch.Tensor, robot_rot: torch.Tensor) -> torch.Tensor:
    delta = torch.matmul(ref_rot, robot_rot.transpose(-1, -2))
    return torch.sum(torch.square(_rotation_vector_from_matrix(delta)), dim=-1)


def _matrix_from_obs_6d(first_two_columns: torch.Tensor) -> torch.Tensor:
    rot_6d = first_two_columns.reshape(*first_two_columns.shape[:-1], 3, 2)
    x_axis = rot_6d[..., :, 0]
    y_axis = rot_6d[..., :, 1]
    x_axis = x_axis / torch.clamp(torch.linalg.norm(x_axis, dim=-1, keepdim=True), min=1e-6)
    y_axis = y_axis - torch.sum(x_axis * y_axis, dim=-1, keepdim=True) * x_axis
    y_axis = y_axis / torch.clamp(torch.linalg.norm(y_axis, dim=-1, keepdim=True), min=1e-6)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def _get_obs_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    if body_names is None:
        return list(range(len(command.cfg.body_names)))
    return [command.cfg.body_names.index(name) for name in body_names]


def _future_ee_reference_from_obs_layout(
    env: ManagerBasedRlEnv,
    command: MotionCommand,
    body_names: list[str],
) -> dict[str, torch.Tensor]:
    """Reconstruct EE reference targets from the same future terms used by EE observations."""
    body_indexes = [command.cfg.body_names.index(name) for name in body_names]
    future_body_pos_w, future_body_quat_w = command.future_body_pos_quat_w(body_indexes)
    num_envs, future_steps, num_bodies, _ = future_body_pos_w.shape

    future_anchor_pos_w = command.motion_anchor_pos.view(num_envs, future_steps, 3)
    future_anchor_quat_w = command.motion_anchor_quat.view(num_envs, future_steps, 4)

    robot_anchor_pos_w = command.robot_anchor_pos_w
    robot_anchor_quat_w = command.robot_anchor_quat_w
    robot_anchor_pos_body = robot_anchor_pos_w[:, None, None, :].expand(-1, future_steps, num_bodies, -1)
    robot_anchor_quat_body = robot_anchor_quat_w[:, None, None, :].expand(-1, future_steps, num_bodies, -1)
    body_pos_b, body_quat_b = subtract_frame_transforms(
        robot_anchor_pos_body,
        robot_anchor_quat_body,
        future_body_pos_w,
        future_body_quat_w,
    )

    robot_anchor_pos_anchor = robot_anchor_pos_w[:, None, :].expand(-1, future_steps, -1)
    robot_anchor_quat_anchor = robot_anchor_quat_w[:, None, :].expand(-1, future_steps, -1)
    anchor_pos_b, anchor_quat_b = subtract_frame_transforms(
        robot_anchor_pos_anchor,
        robot_anchor_quat_anchor,
        future_anchor_pos_w,
        future_anchor_quat_w,
    )

    body_rot_b = matrix_from_quat(body_quat_b)
    anchor_rot_b = matrix_from_quat(anchor_quat_b)
    robot_anchor_rot_w = matrix_from_quat(robot_anchor_quat_w)
    ref_anchor_rot_w = torch.matmul(robot_anchor_rot_w[:, None, :, :], anchor_rot_b)
    ref_anchor_pos_w = robot_anchor_pos_w[:, None, :] + _matvec(robot_anchor_rot_w[:, None, :, :], anchor_pos_b)

    body_pos_rel_anchor = _matvec(
        anchor_rot_b[:, :, None, :, :].transpose(-1, -2),
        body_pos_b - anchor_pos_b[:, :, None, :],
    )
    body_rot_rel_anchor = torch.matmul(anchor_rot_b[:, :, None, :, :].transpose(-1, -2), body_rot_b)

    body_pos_w = ref_anchor_pos_w[:, :, None, :] + _matvec(ref_anchor_rot_w[:, :, None, :, :], body_pos_rel_anchor)
    body_rot_w = torch.matmul(ref_anchor_rot_w[:, :, None, :, :], body_rot_rel_anchor)

    if future_steps > 1:
        linear_velocity_w = (body_pos_w[:, 1] - body_pos_w[:, 0]) / env.step_dt
        delta_rot = torch.matmul(body_rot_w[:, 1], body_rot_w[:, 0].transpose(-1, -2))
        angular_velocity_w = _rotation_vector_from_matrix(delta_rot) / env.step_dt
    else:
        linear_velocity_w = torch.zeros(num_envs, num_bodies, 3, device=body_pos_w.device)
        angular_velocity_w = torch.zeros(num_envs, num_bodies, 3, device=body_pos_w.device)

    return {
        "body_indexes": torch.tensor(body_indexes, device=body_pos_w.device, dtype=torch.long),
        "body_pos_rel_anchor": body_pos_rel_anchor[:, 0],
        "body_rot_rel_anchor": body_rot_rel_anchor[:, 0],
        "body_pos_w": body_pos_w[:, 0],
        "body_rot_w": body_rot_w[:, 0],
        "body_lin_vel_w": linear_velocity_w,
        "body_ang_vel_w": angular_velocity_w,
    }


def _future_obs_relative_body_reference_w(
    env: ManagerBasedRlEnv,
    command: MotionCommand,
    command_name: str,
    body_names: list[str] | None,
) -> dict[str, torch.Tensor]:
    """Build drift-tolerant body reference from the same terms exposed in EE future observations."""
    obs_body_names = list(command.cfg.body_names) if body_names is None else body_names
    body_indexes = _get_obs_body_indexes(command, body_names)
    num_envs = env.num_envs
    num_bodies = len(body_indexes)

    motion_ee_pos_b = obs_motion_body_pos_b_future(env, command_name, obs_body_names).view(
        num_envs, -1, num_bodies, 3
    )
    motion_ee_ori_b = obs_motion_body_ori_b_future(env, command_name, obs_body_names).view(
        num_envs, -1, num_bodies, 6
    )
    motion_anchor_pos_b = obs_motion_anchor_pos_b_future(env, command_name).view(num_envs, -1, 3)
    motion_anchor_ori_b = obs_motion_anchor_ori_b_future(env, command_name).view(num_envs, -1, 6)
    robot_anchor_pos_w_obs = obs_robot_anchor_pos_w(env, command_name).view(num_envs, 3)
    robot_anchor_ori_w = obs_robot_anchor_ori_w(env, command_name).view(num_envs, 6)
    robot_anchor_rot_w = _matrix_from_obs_6d(robot_anchor_ori_w)
    robot_anchor_quat_w_obs = quat_from_matrix(robot_anchor_rot_w)

    body_pos_b = motion_ee_pos_b[:, 0]
    body_rot_b = _matrix_from_obs_6d(motion_ee_ori_b[:, 0])
    body_quat_b = quat_from_matrix(body_rot_b)
    anchor_pos_b = motion_anchor_pos_b[:, 0]
    anchor_rot_b = _matrix_from_obs_6d(motion_anchor_ori_b[:, 0])
    anchor_quat_b = quat_from_matrix(anchor_rot_b)

    robot_anchor_quat_body = robot_anchor_quat_w_obs[:, None, :].expand(-1, num_bodies, -1)
    body_pos_w = robot_anchor_pos_w_obs[:, None, :] + quat_apply(robot_anchor_quat_body, body_pos_b)
    body_quat_w = quat_mul(robot_anchor_quat_body, body_quat_b)
    anchor_pos_w = robot_anchor_pos_w_obs + quat_apply(robot_anchor_quat_w_obs, anchor_pos_b)
    anchor_quat_w = quat_mul(robot_anchor_quat_w_obs, anchor_quat_b)

    delta_pos_w = robot_anchor_pos_w_obs[:, None, :].expand(-1, num_bodies, -1).clone()
    delta_pos_w[..., 2] = anchor_pos_w[:, None, 2]
    delta_yaw_quat = yaw_quat(quat_mul(robot_anchor_quat_w_obs, quat_inv(anchor_quat_w)))
    delta_yaw_quat_body = delta_yaw_quat[:, None, :].expand(-1, num_bodies, -1)

    body_pos_relative_w = delta_pos_w + quat_apply(delta_yaw_quat_body, body_pos_w - anchor_pos_w[:, None, :])
    body_quat_relative_w = quat_mul(delta_yaw_quat_body, body_quat_w)

    return {
        "body_indexes": torch.tensor(body_indexes, device=body_pos_relative_w.device, dtype=torch.long),
        "body_pos_relative_w": body_pos_relative_w,
        "body_quat_relative_w": body_quat_relative_w,
    }


def reward_cond_on_pfail(rew_fn: Callable) -> Callable:
    def wrapper(env: ManagerBasedRlEnv, pfail_threshold: float, *args) -> torch.Tensor:
        reward = rew_fn(env, *args).float()
        pfail_total = env.command_manager.get_term("motion").metrics["pfail_total"]
        reward = reward * (pfail_total < pfail_threshold).float()
        # print(f"pfail_total: {pfail_total}, reward: {reward}")
        return reward

    return wrapper


def contact_forces_cond_on_pfail(
    env: ManagerBasedRlEnv, pfail_threshold: float, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    return reward_cond_on_pfail(contact_forces)(env, pfail_threshold, threshold, sensor_cfg)


#####################


def motion_global_anchor_position_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)**2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = quat_error_magnitude(
        command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]
    )**2
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = quat_error_magnitude(
        command.body_quat_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]
    )**2
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_future_obs_relative_body_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref = _future_obs_relative_body_reference_w(env, command, command_name, body_names)
    body_indexes = ref["body_indexes"]
    error = torch.sum(
        torch.square(ref["body_pos_relative_w"] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_future_obs_relative_body_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref = _future_obs_relative_body_reference_w(env, command, command_name, body_names)
    body_indexes = ref["body_indexes"]
    error = quat_error_magnitude(
        ref["body_quat_relative_w"], command.robot_body_quat_w[:, body_indexes]
    )**2
    return torch.exp(-error.mean(-1) / std**2)


def motion_future_obs_global_body_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str]
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref = _future_ee_reference_from_obs_layout(env, command, body_names)
    body_indexes = ref["body_indexes"]
    error = torch.sum(torch.square(ref["body_pos_w"] - command.robot_body_pos_w[:, body_indexes]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_future_obs_global_body_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str]
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref = _future_ee_reference_from_obs_layout(env, command, body_names)
    body_indexes = ref["body_indexes"]
    robot_body_rot_w = matrix_from_quat(command.robot_body_quat_w[:, body_indexes])
    error = _rotation_error_squared(ref["body_rot_w"], robot_body_rot_w)
    return torch.exp(-error.mean(-1) / std**2)


def motion_future_obs_global_body_linear_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str]
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref = _future_ee_reference_from_obs_layout(env, command, body_names)
    body_indexes = ref["body_indexes"]
    error = torch.sum(torch.square(ref["body_lin_vel_w"] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def motion_future_obs_global_body_angular_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, body_names: list[str]
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref = _future_ee_reference_from_obs_layout(env, command, body_names)
    body_indexes = ref["body_indexes"]
    error = torch.sum(torch.square(ref["body_ang_vel_w"] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


#####################


def feet_contact_time(env: ManagerBasedRlEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    data = contact_sensor.data
    assert data.current_air_time is not None and data.last_contact_time is not None
    first_air = data.current_air_time[:, sensor_cfg.body_ids] <= env.step_dt + 1e-6
    last_contact_time = data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def feet_slide(
    env: ManagerBasedRlEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    data = contact_sensor.data
    if data.force_history is not None:
        contacts = data.force_history[:, sensor_cfg.body_ids].norm(dim=-1).amax(dim=-1) > 1.0
    else:
        assert data.force is not None
        contacts = data.force[:, sensor_cfg.body_ids].norm(dim=-1) > 1.0
    asset: Entity = env.scene[asset_cfg.name]
    body_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]

    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    env.extras["log"]["Metrics/reward/feet_slide_velocity_sum"] = reward
    return reward


def feet_slide_cond_on_pfail(
    env: ManagerBasedRlEnv, pfail_threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    return reward_cond_on_pfail(feet_slide)(env, pfail_threshold, sensor_cfg, asset_cfg)


def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    assert contact_sensor.data.force is not None
    forces = contact_sensor.data.force[:, sensor_cfg.body_ids]
    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]

    if contact_sensor.data.current_contact_time is not None:
        first_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= env.step_dt + 1e-6
    else:
        first_contact = force_magnitude > 1.0
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]

    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/reward/landing_force_mean"] = mean_landing_force
    return cost


# soft_landing_cond_on_pfail = reward_cond_on_pfail(soft_landing)
def soft_landing_cond_on_pfail(
    env: ManagerBasedRlEnv, pfail_threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    return reward_cond_on_pfail(soft_landing)(env, pfail_threshold, sensor_cfg)


def joint_vel_out_of_manual_limit_reward(
    env: ManagerBasedRlEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # compute any violations
    rew = torch.sum((torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity).float(), dim=1)

    env.extras["log"]["Metrics/reward/overspeed"] = rew
    return rew


def joint_vel_out_of_manual_limit_cond_on_pfail_reward(
    env: ManagerBasedRlEnv,
    pfail_threshold: float,
    max_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return reward_cond_on_pfail(joint_vel_out_of_manual_limit_reward)(env, pfail_threshold, max_velocity, asset_cfg)


def joint_effort_out_of_limit_fixed_reward(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = torch.logical_not(
        torch.isclose(
            asset.data.qfrc_actuator[:, asset_cfg.joint_ids], asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
        )
    ).float()
    rew = torch.sum(out_of_limits, dim=1)
    env.extras["log"]["Metrics/reward/overeffort"] = rew
    return rew


def joint_effort_out_of_limit_fixed_cond_on_pfail_reward(
    env: ManagerBasedRlEnv, pfail_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return reward_cond_on_pfail(joint_effort_out_of_limit_fixed_reward)(env, pfail_threshold, asset_cfg)
