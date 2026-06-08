from __future__ import annotations

import numpy as np

import latent_tracker  # noqa: F401
from latent_tracker.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from latent_tracker.tasks.tracking.mdp import MotionCommandCfg
from latent_tracker.tasks.tracking.mdp.commands_multi import MultiMotionLoader
from latent_tracker.tasks.tracking.tracking_env_cfg import TRACKED_BODY_NAMES
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks


def test_g1_cfg_uses_dict_managers_and_action_dim() -> None:
    cfg = G1FlatEnvCfg()
    assert isinstance(cfg.observations, dict)
    assert isinstance(cfg.actions, dict)
    assert isinstance(cfg.commands, dict)
    assert isinstance(cfg.events, dict)
    assert isinstance(cfg.rewards, dict)
    assert isinstance(cfg.terminations, dict)
    assert "actor" in cfg.observations
    assert "critic" in cfg.observations

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    assert action.entity_name == "robot"
    assert action.actuator_names == (".*",)


def test_g1_motion_command_contract() -> None:
    cfg = G1FlatEnvCfg()
    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    assert command.entity_name == "robot"
    assert command.anchor_body_name == "torso_link"
    assert tuple(command.body_names) == TRACKED_BODY_NAMES
    assert command.future_steps == 5
    assert command.freeze_frame_aug is True


def test_motion_data_body_mapping_contract() -> None:
    data = np.load("/home/humanoid/yzh/TextOp/motion_accloss2.npz")
    assert data["joint_pos"].shape[1] == 29
    assert data["body_pos_w"].shape[1:] == (30, 3)
    assert data["body_quat_w"].shape[1:] == (30, 4)
    body_names = [str(name) for name in data["body_names"]]
    body_indexes = [body_names.index(name) for name in TRACKED_BODY_NAMES]
    assert body_indexes == [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]


def test_motion_loader_reorders_joints_to_robot_order() -> None:
    motion_file = "/home/humanoid/yzh/TextOp/motion_accloss2.npz"
    robot_joint_names = [
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
    data = np.load(motion_file)
    motion_joint_names = [str(name) for name in data["joint_names"]]
    reindex = [motion_joint_names.index(name) for name in robot_joint_names]

    loader = MultiMotionLoader(
        [motion_file],
        body_indexes=list(range(len(TRACKED_BODY_NAMES))),
        body_names=TRACKED_BODY_NAMES,
        joint_names=robot_joint_names,
        device="cpu",
    )

    assert reindex != list(range(len(reindex)))
    np.testing.assert_allclose(loader.joint_pos_list[0].cpu().numpy(), data["joint_pos"][:, reindex])
    np.testing.assert_allclose(loader.joint_vel_list[0].cpu().numpy(), data["joint_vel"][:, reindex])


def test_registers_all_g1_latent_tracker_variants() -> None:
    tasks = [task for task in list_tasks() if task.startswith("Mjlab-LatentTracker-Flat-G1")]
    assert len(tasks) == 17
