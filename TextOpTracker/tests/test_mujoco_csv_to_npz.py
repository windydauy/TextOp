from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_csv_to_npz.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mujoco_csv_to_npz_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_argparser_supports_headless_flag():
    module = load_module()

    args = module.build_argparser().parse_args(["--input_file", "in.csv", "--output_name", "out.npz"])
    assert args.headless is False

    args = module.build_argparser().parse_args(["--input_file", "in.csv", "--output_name", "out.npz", "--headless"])
    assert args.headless is True


def test_main_skips_visualization_when_headless(monkeypatch, capsys):
    module = load_module()
    artifact = SimpleNamespace(output_path=Path("/tmp/fake_motion.npz"))
    calls = {"visualize": 0}

    monkeypatch.setattr(module, "convert_csv_to_npz", lambda **kwargs: artifact)
    monkeypatch.setattr(module, "visualize_motion", lambda *args, **kwargs: calls.__setitem__("visualize", calls["visualize"] + 1))
    monkeypatch.setattr(
        sys,
        "argv",
        ["mujoco_csv_to_npz.py", "--input_file", "in.csv", "--output_name", "out.npz", "--headless"],
    )

    module.main()

    assert calls["visualize"] == 0
    assert "[DONE] saved /tmp/fake_motion.npz" in capsys.readouterr().out


def test_main_visualizes_by_default(monkeypatch):
    module = load_module()
    artifact = SimpleNamespace(output_path=Path("/tmp/fake_motion.npz"))
    calls = {"visualize": 0}

    monkeypatch.setattr(module, "convert_csv_to_npz", lambda **kwargs: artifact)
    monkeypatch.setattr(module, "visualize_motion", lambda *args, **kwargs: calls.__setitem__("visualize", calls["visualize"] + 1))
    monkeypatch.setattr(sys, "argv", ["mujoco_csv_to_npz.py", "--input_file", "in.csv", "--output_name", "out.npz"])

    module.main()

    assert calls["visualize"] == 1


def test_visualize_motion_replays_each_frame(monkeypatch):
    module = load_module()
    frames = []

    class FakeModel:
        def __init__(self):
            self.njnt = 3
            self.jnt_type = [0, 0, 0]
            self.jnt_qposadr = [7, 8, 9]
            self.jnt_dofadr = [6, 7, 8]

    class FakeData:
        def __init__(self, model):
            self.qpos = np.zeros(10, dtype=np.float64)
            self.qvel = np.zeros(9, dtype=np.float64)

    class FakeViewer:
        def __init__(self):
            self.cam = SimpleNamespace(lookat=np.zeros(3), distance=None, azimuth=None, elevation=None)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def sync(self):
            frames.append((fake_data.qpos.copy(), fake_data.qvel.copy()))

        def is_running(self):
            return True

    fake_model = FakeModel()
    fake_data = FakeData(fake_model)
    fake_mujoco = SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_path=lambda path: fake_model),
        MjData=lambda model: fake_data,
        mjtObj=SimpleNamespace(mjOBJ_JOINT=0),
        mjtJoint=SimpleNamespace(mjJNT_FREE=99),
        mj_id2name=lambda model, obj_type, joint_id: ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"][joint_id],
        mj_forward=lambda model, data: None,
        viewer=SimpleNamespace(launch_passive=lambda model, data: FakeViewer()),
    )

    monkeypatch.setattr(module, "_load_mujoco", lambda: fake_mujoco)
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    artifact = SimpleNamespace(
        fps=50,
        root_pos=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        root_quat=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
        root_lin_vel=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        root_ang_vel=np.array([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype=np.float64),
        joint_pos=np.array([[0.01, 0.02, 0.03], [0.11, 0.12, 0.13]], dtype=np.float64),
        joint_vel=np.array([[0.21, 0.22, 0.23], [0.31, 0.32, 0.33]], dtype=np.float64),
    )

    module.visualize_motion(artifact, "fake.xml", joint_names=["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"])

    assert len(frames) == 2
    assert np.allclose(frames[0][0][:10], [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.03])
    assert np.allclose(frames[1][0][:10], [4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0, 0.11, 0.12, 0.13])
    assert np.allclose(frames[0][1][:9], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.21, 0.22, 0.23])
    assert np.allclose(frames[1][1][:9], [0.4, 0.5, 0.6, 1.0, 1.1, 1.2, 0.31, 0.32, 0.33])


def test_runtime_joint_names_use_isaac_runtime_order():
    module = load_module()

    assert module.ISAAC_RUNTIME_JOINT_NAMES == [
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


def test_reorder_joint_series_to_runtime_order():
    module = load_module()
    joint_series = np.arange(2 * 29, dtype=np.float64).reshape(2, 29)

    reordered = module.reorder_joint_series_to_runtime_order(joint_series)

    expected_order = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
    assert np.array_equal(reordered, joint_series[:, expected_order])


def test_linear_velocity_is_derived_from_positions():
    module = load_module()
    body_pos_w = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 2.0, 3.0], [3.0, 0.0, 0.0]],
            [[2.0, 4.0, 6.0], [5.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    body_lin_vel_w = module.compute_body_linear_velocities(body_pos_w, fps=2)

    expected = np.gradient(body_pos_w, 0.5, axis=0)
    assert np.allclose(body_lin_vel_w, expected)


def test_angular_velocity_is_derived_from_quaternions_with_sign_continuity():
    module = load_module()
    body_quat_w = np.array(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[-0.92387953, 0.0, 0.0, -0.38268343]],
            [[0.70710678, 0.0, 0.0, 0.70710678]],
        ],
        dtype=np.float64,
    )

    body_ang_vel_w = module.compute_body_angular_velocities(body_quat_w, fps=2)

    assert np.max(np.abs(body_ang_vel_w[..., :2])) < 1e-6
    assert np.all(body_ang_vel_w[..., 2] > 0.0)
    assert np.max(np.abs(body_ang_vel_w[..., 2])) < 4.0
