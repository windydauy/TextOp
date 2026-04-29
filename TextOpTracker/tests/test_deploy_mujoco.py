from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "deploy_mujoco.py"


def load_tree() -> ast.AST:
    return ast.parse(SCRIPT_PATH.read_text(), filename=str(SCRIPT_PATH))


def _install_fake_runtime_modules() -> None:
    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.viewer = ModuleType("mujoco.viewer")
    fake_mujoco._enums = SimpleNamespace(mjtGeom=type("FakeGeomEnum", (), {}))
    fake_mujoco.mjtObj = SimpleNamespace(mjOBJ_UNKNOWN=0)
    fake_mujoco.mjtCatBit = SimpleNamespace(mjCAT_DECOR=0)
    fake_mujoco.mjtGeom = SimpleNamespace(mjGEOM_BOX=0, mjGEOM_SPHERE=1)

    fake_mujoco_viewer = ModuleType("mujoco_viewer")
    fake_mujoco_viewer.MujocoViewer = type(
        "FakeViewer", (), {"_add_marker_to_scene": staticmethod(lambda *_a, **_k: None)}
    )

    fake_ort = ModuleType("onnxruntime")

    sys.modules["mujoco"] = fake_mujoco
    sys.modules["mujoco.viewer"] = fake_mujoco.viewer
    sys.modules["mujoco_viewer"] = fake_mujoco_viewer
    sys.modules["onnxruntime"] = fake_ort


def load_module():
    _install_fake_runtime_modules()
    spec = importlib.util.spec_from_file_location("deploy_mujoco_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_motion_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        fps=np.array([50], dtype=np.int64),
        joint_pos=np.zeros((2, 29), dtype=np.float32),
        joint_vel=np.zeros((2, 29), dtype=np.float32),
        body_pos_w=np.zeros((2, 14, 3), dtype=np.float32),
        body_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 14, 1)),
    )


def test_deploy_parser_supports_anchor_body_name_flag():
    tree = load_tree()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--anchor_body_name":
            continue

        keyword_values = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in node.keywords
            if keyword.arg in {"default", "choices"}
        }
        assert keyword_values["default"] == "pelvis"
        assert keyword_values["choices"] == ["pelvis", "torso_link"]
        return

    raise AssertionError("Expected deploy_mujoco.py to register a --anchor_body_name flag")


def test_deploy_parser_supports_debug_obs_flags():
    tree = load_tree()
    found_flags = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        option = node.args[0].value
        if option == "--debug_obs":
            keyword_values = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg in {"action", "default"}
            }
            assert keyword_values["action"] == "store_true"
            assert keyword_values["default"] is False
            found_flags.add(option)
        elif option == "--debug_obs_steps":
            keyword_values = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg in {"default"}
            }
            assert keyword_values["default"] == 5
            found_flags.add(option)

    assert found_flags == {"--debug_obs", "--debug_obs_steps"}


def test_motion_loader_accepts_explicit_anchor_body_name(tmp_path):
    module = load_module()
    motion_path = tmp_path / "motion.npz"
    _write_motion_npz(motion_path)

    loader = module.MotionLoader(motion_path, anchor_body_name="torso_link")

    assert loader.anchor_body_name == "torso_link"
    assert loader.anchor_body_index == loader.body_names.index("torso_link")


def test_motion_loader_rejects_unknown_anchor_body_name(tmp_path):
    module = load_module()
    motion_path = tmp_path / "motion.npz"
    _write_motion_npz(motion_path)

    with pytest.raises(ValueError, match="Unsupported anchor_body_name"):
        module.MotionLoader(motion_path, anchor_body_name="head")


def test_iter_observation_segments_matches_proj_grav_anchor_obs_layout():
    module = load_module()
    obs = np.arange(module._OBS_DIMS["ProjGravAnchorObs"], dtype=np.float32)

    segments = list(module.iter_observation_segments(obs, "ProjGravAnchorObs"))

    assert [name for name, _ in segments] == [
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
    ]
    assert [segment.shape[0] for _, segment in segments] == [290, 15, 30, 3, 6, 3, 3, 3, 29, 29, 29]
    assert np.array_equal(segments[0][1][:3], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert np.array_equal(segments[-1][1][-3:], np.array([437.0, 438.0, 439.0], dtype=np.float32))
