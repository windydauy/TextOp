from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_SRC = REPO_ROOT / "TextOpTracker" / "source" / "textop_tracker"


def _load_mapping_module():
    module_path = (
        TRACKER_SRC
        / "textop_tracker"
        / "tasks"
        / "tracking"
        / "mdp"
        / "motion_body_index.py"
    )
    spec = importlib.util.spec_from_file_location("motion_body_index", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_motion_body_indexes_uses_npz_body_names() -> None:
    module = _load_mapping_module()
    motion_path = (
        REPO_ROOT
        / "motion_ae"
        / "optitrack_npz_filtered_all"
        / "BALANCE_029_Skeleton 006_z_up_x_forward_gym"
        / "motion.npz"
    )
    with np.load(motion_path) as data:
        motion_body_names = data["body_names"]

    requested_body_names = [
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

    indexes = module.resolve_motion_body_indexes(
        motion_body_names=motion_body_names,
        requested_body_names=requested_body_names,
        fallback_body_indexes=[999] * len(requested_body_names),
        motion_file=str(motion_path),
    )

    assert indexes == [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]


def test_resolve_motion_body_indexes_falls_back_for_legacy_npz_without_names() -> None:
    module = _load_mapping_module()

    indexes = module.resolve_motion_body_indexes(
        motion_body_names=None,
        requested_body_names=["pelvis", "torso_link"],
        fallback_body_indexes=[2, 7],
        motion_file="legacy_motion.npz",
    )

    assert indexes == [2, 7]
