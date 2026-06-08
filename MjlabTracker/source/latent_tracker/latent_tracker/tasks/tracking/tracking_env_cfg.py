from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    ActionTermCfg,
    CommandTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import latent_tracker.tasks.tracking.mdp as mdp
from latent_tracker.tasks.tracking.mdp import MotionCommandCfg

DEFAULT_MOTION_FILE = str(Path("/home/humanoid/yzh/TextOp/motion_accloss2.npz"))

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

EE_BODY_NAMES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

NON_EE_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "right_hip_roll_link",
    "right_knee_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
)

TRACKED_BODY_NAMES = (
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
)

NON_EE_BODY_REWARD_WEIGHT = len(NON_EE_BODY_NAMES) / len(TRACKED_BODY_NAMES)
EE_BODY_REWARD_WEIGHT = len(EE_BODY_NAMES) / len(TRACKED_BODY_NAMES)

ObservationMode = Literal[
    "base",
    "proj_grav",
    "proj_grav_anchor",
    "proj_grav_anchor_ee",
    "prop_prop",
    "priv_priv",
    "noise_priv",
]
RewardMode = Literal["base", "transformer_vae_ee", "split_body_transformer_vae_ee"]


def _obs_group(terms: dict[str, ObservationTermCfg], *, corruption: bool) -> ObservationGroupCfg:
    return ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=corruption,
    )


def _actor_terms(mode: ObservationMode) -> dict[str, ObservationTermCfg]:
    privileged = mode in {"priv_priv", "noise_priv"}
    terms: dict[str, ObservationTermCfg] = {
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "motion"}),
        "motion_anchor_pos_b": ObservationTermCfg(
            func=mdp.motion_anchor_pos_b_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        ),
        "motion_anchor_ori_b": ObservationTermCfg(
            func=mdp.motion_anchor_ori_b_future,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
    }
    if mode in {"proj_grav_anchor", "proj_grav_anchor_ee"}:
        terms.update(
            {
                "robot_anchor_pos_w": ObservationTermCfg(
                    func=mdp.robot_anchor_pos_w,
                    params={"command_name": "motion"},
                    noise=Unoise(n_min=-0.05, n_max=0.05),
                ),
                "robot_anchor_ori_w": ObservationTermCfg(
                    func=mdp.robot_anchor_ori_w,
                    params={"command_name": "motion"},
                    noise=Unoise(n_min=-0.05, n_max=0.05),
                ),
            }
        )
    if mode == "proj_grav_anchor_ee":
        terms.update(
            {
                "motion_ee_pos_b": ObservationTermCfg(
                    func=mdp.motion_body_pos_b_future,
                    params={"command_name": "motion", "body_names": list(EE_BODY_NAMES)},
                    noise=Unoise(n_min=-0.1, n_max=0.1),
                ),
                "motion_ee_ori_b": ObservationTermCfg(
                    func=mdp.motion_body_ori_b_future,
                    params={"command_name": "motion", "body_names": list(EE_BODY_NAMES)},
                    noise=Unoise(n_min=-0.05, n_max=0.05),
                ),
            }
        )
    if mode in {"proj_grav", "proj_grav_anchor", "proj_grav_anchor_ee"}:
        terms["projected_gravity"] = ObservationTermCfg(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.07, n_max=0.07)
        )
    if privileged:
        terms.update(
            {
                "body_pos": ObservationTermCfg(
                    func=mdp.robot_body_pos_b,
                    params={"command_name": "motion"},
                    noise=Unoise(n_min=-0.2, n_max=0.2) if mode == "noise_priv" else None,
                ),
                "body_ori": ObservationTermCfg(
                    func=mdp.robot_body_ori_b,
                    params={"command_name": "motion"},
                    noise=Unoise(n_min=-0.1, n_max=0.1) if mode == "noise_priv" else None,
                ),
            }
        )
    terms.update(
        {
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)),
            "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)),
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
            "actions": ObservationTermCfg(func=mdp.last_action),
        }
    )
    return terms


def _critic_terms(mode: ObservationMode) -> dict[str, ObservationTermCfg]:
    if mode == "prop_prop":
        return deepcopy(_actor_terms("base"))
    privileged_mode: ObservationMode = "noise_priv" if mode == "noise_priv" else "priv_priv"
    terms = _actor_terms(privileged_mode)
    if mode in {"proj_grav", "proj_grav_anchor", "proj_grav_anchor_ee"}:
        insert = {"projected_gravity": ObservationTermCfg(func=mdp.projected_gravity)}
        rebuilt: dict[str, ObservationTermCfg] = {}
        for key, value in terms.items():
            rebuilt[key] = value
            if key == "motion_anchor_ori_b":
                rebuilt.update(insert)
        terms = rebuilt
    if mode in {"proj_grav_anchor", "proj_grav_anchor_ee"}:
        anchor = {
            "robot_anchor_pos_w": ObservationTermCfg(func=mdp.robot_anchor_pos_w, params={"command_name": "motion"}),
            "robot_anchor_ori_w": ObservationTermCfg(func=mdp.robot_anchor_ori_w, params={"command_name": "motion"}),
        }
        rebuilt = {}
        for key, value in terms.items():
            rebuilt[key] = value
            if key == "motion_anchor_ori_b":
                rebuilt.update(anchor)
        terms = rebuilt
    if mode == "proj_grav_anchor_ee":
        ee = {
            "motion_ee_pos_b": ObservationTermCfg(
                func=mdp.motion_body_pos_b_future,
                params={"command_name": "motion", "body_names": list(EE_BODY_NAMES)},
            ),
            "motion_ee_ori_b": ObservationTermCfg(
                func=mdp.motion_body_ori_b_future,
                params={"command_name": "motion", "body_names": list(EE_BODY_NAMES)},
            ),
        }
        rebuilt = {}
        for key, value in terms.items():
            rebuilt[key] = value
            if key == "robot_anchor_ori_w":
                rebuilt.update(ee)
        terms = rebuilt
    return terms


def make_observations_cfg(mode: ObservationMode = "base") -> dict[str, ObservationGroupCfg]:
    actor_corruption = mode != "priv_priv"
    return {
        "actor": _obs_group(_actor_terms(mode), corruption=actor_corruption),
        "critic": _obs_group(_critic_terms(mode), corruption=mode == "noise_priv"),
    }


def make_rewards_cfg(mode: RewardMode = "base") -> dict[str, RewardTermCfg]:
    rewards: dict[str, RewardTermCfg] = {
        "motion_global_anchor_pos": RewardTermCfg(
            func=mdp.motion_global_anchor_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_global_anchor_ori": RewardTermCfg(
            func=mdp.motion_global_anchor_orientation_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_pos": RewardTermCfg(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_body_ori": RewardTermCfg(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_lin_vel": RewardTermCfg(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 1.0},
        ),
        "motion_body_ang_vel": RewardTermCfg(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 3.14},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
        "joint_limit": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        ),
        "feet_force": RewardTermCfg(
            func=mdp.contact_forces_cond_on_pfail,
            weight=-0.0,
            params={
                "sensor_cfg": SceneEntityCfg("feet_contact"),
                "threshold": 600,
                "pfail_threshold": 1.0,
            },
        ),
        "feet_slide": RewardTermCfg(
            func=mdp.feet_slide_cond_on_pfail,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("feet_contact"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
                "pfail_threshold": 0.2,
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing_cond_on_pfail,
            weight=-1e-5,
            params={"sensor_cfg": SceneEntityCfg("feet_contact"), "pfail_threshold": 0.2},
        ),
        "overspeed": RewardTermCfg(
            func=mdp.joint_vel_out_of_manual_limit_cond_on_pfail_reward,
            weight=-0.1,
            params={"max_velocity": 20.0, "pfail_threshold": 0.15},
        ),
        "overeffort": RewardTermCfg(
            func=mdp.joint_effort_out_of_limit_fixed_cond_on_pfail_reward,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "pfail_threshold": 0.15},
        ),
    }
    if mode == "transformer_vae_ee":
        for key in ("feet_slide", "soft_landing", "overspeed", "overeffort"):
            rewards.pop(key, None)
        rewards["motion_ee_global_body_pos"] = RewardTermCfg(
            func=mdp.motion_global_body_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3, "body_names": list(EE_BODY_NAMES)},
        )
        rewards["motion_ee_global_body_ori"] = RewardTermCfg(
            func=mdp.motion_global_body_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4, "body_names": list(EE_BODY_NAMES)},
        )
    elif mode == "split_body_transformer_vae_ee":
        for key in ("feet_slide", "soft_landing", "overspeed", "overeffort"):
            rewards.pop(key, None)
        rewards["motion_body_pos"] = RewardTermCfg(
            func=mdp.motion_relative_body_position_error_exp,
            weight=NON_EE_BODY_REWARD_WEIGHT,
            params={"command_name": "motion", "std": 0.3, "body_names": list(NON_EE_BODY_NAMES)},
        )
        rewards["motion_body_ori"] = RewardTermCfg(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=NON_EE_BODY_REWARD_WEIGHT,
            params={"command_name": "motion", "std": 0.4, "body_names": list(NON_EE_BODY_NAMES)},
        )
        rewards["motion_body_lin_vel"] = RewardTermCfg(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=NON_EE_BODY_REWARD_WEIGHT,
            params={"command_name": "motion", "std": 1.0, "body_names": list(NON_EE_BODY_NAMES)},
        )
        rewards["motion_body_ang_vel"] = RewardTermCfg(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=NON_EE_BODY_REWARD_WEIGHT,
            params={"command_name": "motion", "std": 3.14, "body_names": list(NON_EE_BODY_NAMES)},
        )
        rewards["motion_ee_body_pos"] = RewardTermCfg(
            func=mdp.motion_future_obs_relative_body_position_error_exp,
            weight=EE_BODY_REWARD_WEIGHT,
            params={"command_name": "motion", "std": 0.3, "body_names": list(EE_BODY_NAMES)},
        )
        rewards["motion_ee_body_ori"] = RewardTermCfg(
            func=mdp.motion_future_obs_relative_body_orientation_error_exp,
            weight=EE_BODY_REWARD_WEIGHT,
            params={"command_name": "motion", "std": 0.4, "body_names": list(EE_BODY_NAMES)},
        )
        rewards["motion_ee_body_lin_vel"] = RewardTermCfg(
            func=mdp.motion_future_obs_global_body_linear_velocity_error_exp,
            weight=EE_BODY_REWARD_WEIGHT,
            params={"command_name": "motion", "std": 1.0, "body_names": list(EE_BODY_NAMES)},
        )
        rewards["motion_ee_body_ang_vel"] = RewardTermCfg(
            func=mdp.motion_future_obs_global_body_angular_velocity_error_exp,
            weight=EE_BODY_REWARD_WEIGHT,
            params={"command_name": "motion", "std": 3.14, "body_names": list(EE_BODY_NAMES)},
        )
        rewards["motion_ee_global_body_pos"] = RewardTermCfg(
            func=mdp.motion_future_obs_global_body_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3, "body_names": list(EE_BODY_NAMES)},
        )
        rewards["motion_ee_global_body_ori"] = RewardTermCfg(
            func=mdp.motion_future_obs_global_body_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4, "body_names": list(EE_BODY_NAMES)},
        )
    return rewards


def make_tracking_env_cfg(
    *,
    observation_mode: ObservationMode = "base",
    reward_mode: RewardMode = "base",
    motion_files: list[str] | None = None,
    num_envs: int = 4096,
) -> ManagerBasedRlEnvCfg:
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }
    commands: dict[str, CommandTermCfg] = {
        "motion": MotionCommandCfg(
            entity_name="robot",
            motion_files=motion_files or [DEFAULT_MOTION_FILE],
            anchor_body_name="",
            body_names=[],
            future_steps=5,
            freeze_frame_aug=True,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range=VELOCITY_RANGE,
            joint_position_range=(-0.1, 0.1),
        )
    }
    events: dict[str, EventTermCfg] = {
        "add_joint_default_pos": EventTermCfg(
            func=mdp.randomize_joint_default_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "pos_distribution_params": (-0.01, 0.01),
                "operation": "add",
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        ),
        "randomize_rigid_body_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*wrist_yaw.*|torso_link"),
                "ranges": (0.8, 2.5),
                "operation": "scale",
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": VELOCITY_RANGE},
        ),
    }
    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "anchor_pos": TerminationTermCfg(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25},
        ),
        "anchor_ori": TerminationTermCfg(
            func=mdp.bad_anchor_ori,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
        ),
        "ee_body_pos": TerminationTermCfg(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": list(EE_BODY_NAMES),
            },
        ),
    }

    feet_contact_cfg = ContactSensorCfg(
        name="feet_contact",
        primary=ContactMatch(mode="body", pattern=EE_BODY_NAMES[-2:], entity="robot"),
        secondary=None,
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
        global_frame=True,
        history_length=3,
    )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=num_envs,
            env_spacing=2.5,
            sensors=(feet_contact_cfg,),
        ),
        observations=make_observations_cfg(observation_mode),
        actions=actions,
        commands=commands,
        events=events,
        rewards=make_rewards_cfg(reward_mode),
        terminations=terminations,
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",
            distance=2.8,
            fovy=55.0,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=250,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=10.0,
        scale_rewards_by_dt=False,
    )


TrackingEnvCfg = make_tracking_env_cfg
