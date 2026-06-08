from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def randomize_joint_default_pos(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Randomize qpos0 and keep the position action offset aligned."""
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    if pos_distribution_params is None:
        return

    dr.joint_default_pos(
        env,
        env_ids,
        ranges=pos_distribution_params,
        asset_cfg=asset_cfg,
        operation=operation,
        distribution=distribution,
    )

    action_term = env.action_manager.get_term("joint_pos")
    if hasattr(action_term, "_offset"):
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            action_term._offset[env_ids] = asset.data.default_joint_pos[env_ids]
        else:
            joint_ids_tensor = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
            action_term._offset[env_ids[:, None], joint_ids_tensor] = asset.data.default_joint_pos[
                env_ids[:, None], joint_ids_tensor
            ]


def randomize_rigid_body_com(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    ranges = {
        0: com_range.get("x", (0.0, 0.0)),
        1: com_range.get("y", (0.0, 0.0)),
        2: com_range.get("z", (0.0, 0.0)),
    }
    dr.body_com_offset(
        env,
        env_ids,
        ranges=ranges,
        asset_cfg=asset_cfg,
        operation="add",
    )


def push_by_setting_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Set random root linear/angular velocity, matching the IsaacLab push event semantics."""
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)

    ranges = [
        velocity_range.get(axis, (0.0, 0.0))
        for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    ]
    low = torch.tensor([r[0] for r in ranges], device=env.device)
    high = torch.tensor([r[1] for r in ranges], device=env.device)
    root_vel = sample_uniform(low, high, (len(env_ids), 6), env.device)
    asset.write_root_com_velocity_to_sim(root_vel, env_ids=env_ids)
