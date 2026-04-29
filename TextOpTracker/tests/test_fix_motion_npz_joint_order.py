from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fix_motion_npz_joint_order.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fix_motion_npz_joint_order_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repair_motion_npz_reorders_legacy_joint_axes_and_adds_metadata(tmp_path):
    module = load_module()
    motion_dir = tmp_path / "clip_a"
    motion_dir.mkdir()
    motion_path = motion_dir / "motion.npz"

    joint_pos = np.arange(58, dtype=np.float32).reshape(2, 29)
    joint_vel = joint_pos + 1000.0
    body_pos = np.arange(2 * 30 * 3, dtype=np.float32).reshape(2, 30, 3)
    body_quat = np.arange(2 * 30 * 4, dtype=np.float32).reshape(2, 30, 4)
    body_lin_vel = np.arange(2 * 30 * 3, dtype=np.float32).reshape(2, 30, 3) + 2000.0
    body_ang_vel = np.arange(2 * 30 * 3, dtype=np.float32).reshape(2, 30, 3) + 3000.0
    np.savez(
        motion_path,
        fps=np.array([50], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
    )

    result = module.repair_motion_npz(motion_path)

    assert result.status == "repaired"
    assert motion_path.with_suffix(".npz.bak").exists()

    repaired = np.load(motion_path, allow_pickle=True)
    expected_joint_pos = joint_pos[:, [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]]
    expected_joint_vel = joint_vel[:, [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]]

    assert np.array_equal(repaired["joint_pos"], expected_joint_pos)
    assert np.array_equal(repaired["joint_vel"], expected_joint_vel)
    assert repaired["joint_names"].tolist() == module.ISAAC_RUNTIME_JOINT_NAMES
    assert repaired["body_names"].tolist() == module.ISAAC_G1_BODY_NAMES
    assert np.array_equal(repaired["body_pos_w"], body_pos)
    assert np.array_equal(repaired["body_quat_w"], body_quat)
    assert np.array_equal(repaired["body_lin_vel_w"], body_lin_vel)
    assert np.array_equal(repaired["body_ang_vel_w"], body_ang_vel)


def test_repair_motion_npz_skips_already_fixed_file(tmp_path):
    module = load_module()
    motion_dir = tmp_path / "clip_b"
    motion_dir.mkdir()
    motion_path = motion_dir / "motion.npz"

    np.savez(
        motion_path,
        fps=np.array([50], dtype=np.int64),
        joint_pos=np.zeros((2, 29), dtype=np.float32),
        joint_vel=np.zeros((2, 29), dtype=np.float32),
        body_pos_w=np.zeros((2, 30, 3), dtype=np.float32),
        body_quat_w=np.zeros((2, 30, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((2, 30, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 30, 3), dtype=np.float32),
        joint_names=np.array(module.ISAAC_RUNTIME_JOINT_NAMES, dtype="<U64"),
    )

    result = module.repair_motion_npz(motion_path)

    assert result.status == "skipped"
    assert "already uses Isaac runtime joint order" in result.message
    assert not motion_path.with_suffix(".npz.bak").exists()


def test_repair_motion_tree_repairs_multiple_files(tmp_path):
    module = load_module()
    for idx in range(2):
        motion_dir = tmp_path / f"clip_{idx}"
        motion_dir.mkdir()
        np.savez(
            motion_dir / "motion.npz",
            fps=np.array([50], dtype=np.int64),
            joint_pos=np.arange(29, dtype=np.float32).reshape(1, 29),
            joint_vel=np.arange(29, dtype=np.float32).reshape(1, 29),
            body_pos_w=np.zeros((1, 30, 3), dtype=np.float32),
            body_quat_w=np.zeros((1, 30, 4), dtype=np.float32),
            body_lin_vel_w=np.zeros((1, 30, 3), dtype=np.float32),
            body_ang_vel_w=np.zeros((1, 30, 3), dtype=np.float32),
        )

    results = module.repair_motion_tree(tmp_path)

    assert len(results) == 2
    assert all(result.status == "repaired" for result in results)
