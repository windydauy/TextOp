from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LEGACY_MUJOCO_JOINT_NAMES = [
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

ISAAC_RUNTIME_JOINT_NAMES = [
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

ISAAC_G1_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


@dataclass(frozen=True)
class RepairResult:
    path: Path
    status: str
    message: str = ""


def _as_name_list(array: np.ndarray) -> list[str]:
    return [str(item) for item in np.asarray(array).tolist()]


def _infer_joint_order(npz_data, assume_legacy_when_missing: bool) -> list[str] | None:
    if "joint_names" in npz_data.files:
        return _as_name_list(npz_data["joint_names"])
    if assume_legacy_when_missing:
        return LEGACY_MUJOCO_JOINT_NAMES
    return None


def _reorder_joint_series(joint_series: np.ndarray, current_joint_names: list[str]) -> np.ndarray:
    reindex = [current_joint_names.index(name) for name in ISAAC_RUNTIME_JOINT_NAMES]
    return np.asarray(joint_series)[:, reindex]


def repair_motion_npz(
    motion_path: str | Path,
    *,
    create_backup: bool = True,
    assume_legacy_when_missing: bool = True,
) -> RepairResult:
    motion_path = Path(motion_path)
    with np.load(motion_path, allow_pickle=True) as data:
        current_joint_names = _infer_joint_order(data, assume_legacy_when_missing=assume_legacy_when_missing)
        if current_joint_names is None:
            return RepairResult(motion_path, "skipped", "missing joint_names and legacy fallback disabled")

        if current_joint_names == ISAAC_RUNTIME_JOINT_NAMES:
            return RepairResult(motion_path, "skipped", "already uses Isaac runtime joint order")

        if sorted(current_joint_names) != sorted(ISAAC_RUNTIME_JOINT_NAMES):
            return RepairResult(motion_path, "skipped", "joint_names are not a known G1 permutation")

        repaired = {key: data[key] for key in data.files}
        repaired["joint_pos"] = _reorder_joint_series(data["joint_pos"], current_joint_names).astype(data["joint_pos"].dtype, copy=False)
        repaired["joint_vel"] = _reorder_joint_series(data["joint_vel"], current_joint_names).astype(data["joint_vel"].dtype, copy=False)
        repaired["joint_names"] = np.array(ISAAC_RUNTIME_JOINT_NAMES, dtype="<U64")
        if "body_names" not in repaired:
            repaired["body_names"] = np.array(ISAAC_G1_BODY_NAMES, dtype="<U64")

    if create_backup:
        backup_path = motion_path.with_suffix(motion_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(motion_path, backup_path)

    tmp_path = motion_path.with_suffix(motion_path.suffix + ".tmp")
    np.savez(tmp_path, **repaired)
    tmp_npz_path = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    tmp_npz_path.replace(motion_path)
    return RepairResult(motion_path, "repaired")


def repair_motion_tree(
    root_dir: str | Path,
    *,
    create_backup: bool = True,
    assume_legacy_when_missing: bool = True,
) -> list[RepairResult]:
    root_dir = Path(root_dir)
    return [
        repair_motion_npz(
            motion_path,
            create_backup=create_backup,
            assume_legacy_when_missing=assume_legacy_when_missing,
        )
        for motion_path in sorted(root_dir.glob("*/motion.npz"))
    ]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair legacy motion.npz joint order in place.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/humanoid/yzh/TextOp/motions_lafan",
        help="Directory containing per-motion subdirectories with motion.npz files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Overwrite files without creating motion.npz.bak backups.",
    )
    parser.add_argument(
        "--missing-joint-names-are-legacy",
        action="store_true",
        default=True,
        help="Treat files without joint_names metadata as legacy MuJoCo order.",
    )
    parser.add_argument(
        "--strict-metadata",
        action="store_true",
        help="Skip files that are missing joint_names metadata instead of assuming legacy order.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    assume_legacy_when_missing = args.missing_joint_names_are_legacy and not args.strict_metadata
    results = repair_motion_tree(
        args.root_dir,
        create_backup=not args.no_backup,
        assume_legacy_when_missing=assume_legacy_when_missing,
    )

    repaired = sum(result.status == "repaired" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    for result in results:
        suffix = f" ({result.message})" if result.message else ""
        print(f"[{result.status.upper()}] {result.path}{suffix}")
    print(f"[SUMMARY] repaired={repaired} skipped={skipped} total={len(results)}")


if __name__ == "__main__":
    main()
