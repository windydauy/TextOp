"""Replay a motion from a csv file and export it to motion.npz."""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay motion from a csv file and output it to a npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="frame range: START END (both inclusive). The frame index starts from 1.",
)
parser.add_argument("--output_name", type=str, required=True, help="The output artifact directory name.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument(
    "--root_quat_order",
    type=str,
    default="xyzw",
    choices=("xyzw", "wxyz"),
    help="Quaternion component order in the csv root pose columns.",
)
parser.add_argument(
    "--input_format",
    type=str,
    default="auto",
    choices=("auto", "legacy", "header"),
    help=(
        "CSV input format. 'legacy' expects numeric-only csv with first 7 columns as "
        "[pos(3)+quat(4)]. 'header' expects columns like root_translate* and root_rotate*. "
        "'auto' detects from the first line."
    ),
)
parser.add_argument(
    "--joint_angle_unit",
    type=str,
    default="auto",
    choices=("auto", "rad", "deg"),
    help=(
        "Unit of *_dof joint columns for header csv. 'auto' will infer by magnitude and "
        "convert degrees to radians when needed."
    ),
)
parser.add_argument(
    "--translate_scale",
    type=str,
    default="auto",
    help=(
        "Input unit scale for root translation columns (root_translateX/Y/Z): "
        "'1.0' means the CSV translation is already in metres, '100.0' means "
        "the CSV translation is in centimetres (the script divides by 100). "
        "'auto' infers by magnitude."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from motion_replay_common import G1_JOINT_NAMES, MotionLoader, ReplayMotionsSceneCfg, run_motion_replay


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _euler_xyz_deg_to_quat_wxyz(rx_deg: np.ndarray, ry_deg: np.ndarray, rz_deg: np.ndarray) -> np.ndarray:
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)
    cx, sx = np.cos(rx * 0.5), np.sin(rx * 0.5)
    cy, sy = np.cos(ry * 0.5), np.sin(ry * 0.5)
    cz, sz = np.cos(rz * 0.5), np.sin(rz * 0.5)
    qw = cx * cy * cz - sx * sy * sz
    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    return np.stack([qw, qx, qy, qz], axis=1).astype(np.float32, copy=False)


def _frame_slice(frame_range: tuple[int, int] | None, total_frames: int) -> slice:
    if frame_range is None:
        return slice(None)
    start, end = frame_range
    if start < 1 or end < start:
        raise ValueError(f"Invalid frame_range: {frame_range}. Expected 1-based inclusive START END.")
    start_idx = start - 1
    end_idx_exclusive = min(end, total_frames)
    if start_idx >= total_frames:
        raise ValueError(f"frame_range start {start} exceeds total frames {total_frames}.")
    return slice(start_idx, end_idx_exclusive)


def _apply_translate_scale(base_pos: np.ndarray, translate_scale: str) -> np.ndarray:
    """Convert root translation to metres."""
    if translate_scale == "auto":
        max_abs = float(np.max(np.abs(base_pos)))
        if max_abs > 10.0:
            print(
                f"[INFO] translate_scale=auto: max |translation|={max_abs:.3f} > 10 → "
                "treating as centimetres, dividing by 100 to convert to metres."
            )
            return (base_pos / 100.0).astype(np.float32, copy=False)
        else:
            print(
                f"[INFO] translate_scale=auto: max |translation|={max_abs:.3f} ≤ 10 → "
                "treating as metres, no scaling applied."
            )
            return base_pos
    else:
        unit_scale = float(translate_scale)
        if unit_scale <= 0:
            raise ValueError(f"translate_scale must be > 0, got {unit_scale}.")
        if unit_scale != 1.0:
            print(
                f"[INFO] translate_scale={unit_scale}: interpreting CSV translation unit as 1/{unit_scale} metre, "
                f"dividing by {unit_scale} to convert to metres."
            )
        return (base_pos / unit_scale).astype(np.float32, copy=False)


def _convert_header_csv_to_motion_array(
    motion_file: str | Path,
    frame_range: tuple[int, int] | None,
    joint_angle_unit: str,
    translate_scale: str,
) -> tuple[np.ndarray, str]:
    motion_path = Path(motion_file)
    with motion_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required_pos = ("root_translateX", "root_translateY", "root_translateZ")
        if not all(col in fieldnames for col in required_pos):
            raise ValueError(f"Missing required root translation columns in {motion_path}: {required_pos}")

        has_w = "root_rotateW" in fieldnames
        has_xyz = all(col in fieldnames for col in ("root_rotateX", "root_rotateY", "root_rotateZ"))
        if not has_xyz:
            raise ValueError(
                f"Missing required root rotation columns in {motion_path}. Need root_rotateX/Y/Z at minimum."
            )

        dof_cols = [col for col in fieldnames if col.endswith("_dof")]
        if not dof_cols:
            raise ValueError(f"No *_dof columns found in {motion_path}.")

        rows: list[dict[str, str]] = [row for row in reader if row]
        if not rows:
            raise ValueError(f"No data rows found in {motion_path}.")

    total = len(rows)
    row_slice = _frame_slice(frame_range, total)
    rows = rows[row_slice]

    base_pos = np.stack(
        [
            np.array([float(row["root_translateX"]) for row in rows], dtype=np.float32),
            np.array([float(row["root_translateY"]) for row in rows], dtype=np.float32),
            np.array([float(row["root_translateZ"]) for row in rows], dtype=np.float32),
        ],
        axis=1,
    )

    if has_w:
        # Header contains explicit W: treat as wxyz by default.
        quat_wxyz = np.stack(
            [
                np.array([float(row["root_rotateW"]) for row in rows], dtype=np.float32),
                np.array([float(row["root_rotateX"]) for row in rows], dtype=np.float32),
                np.array([float(row["root_rotateY"]) for row in rows], dtype=np.float32),
                np.array([float(row["root_rotateZ"]) for row in rows], dtype=np.float32),
            ],
            axis=1,
        )
    else:
        quat_wxyz = _euler_xyz_deg_to_quat_wxyz(
            np.array([float(row["root_rotateX"]) for row in rows], dtype=np.float32),
            np.array([float(row["root_rotateY"]) for row in rows], dtype=np.float32),
            np.array([float(row["root_rotateZ"]) for row in rows], dtype=np.float32),
        )

    dof = np.stack(
        [np.array([float(row[col]) for row in rows], dtype=np.float32) for col in dof_cols],
        axis=1,
    )
    if joint_angle_unit == "deg":
        dof = np.deg2rad(dof).astype(np.float32, copy=False)
    elif joint_angle_unit == "auto":
        max_abs = float(np.max(np.abs(dof))) if dof.size > 0 else 0.0
        if max_abs > math.pi * 2.5:
            print(
                f"[INFO] joint_angle_unit=auto: max |dof|={max_abs:.3f} > {math.pi * 2.5:.3f} → "
                "treating as degrees, converting to radians."
            )
            dof = np.deg2rad(dof).astype(np.float32, copy=False)
        else:
            print(
                f"[INFO] joint_angle_unit=auto: max |dof|={max_abs:.3f} ≤ {math.pi * 2.5:.3f} → "
                "treating as radians, no conversion."
            )

    base_pos = _apply_translate_scale(base_pos, translate_scale)

    motion = np.concatenate([base_pos, quat_wxyz, dof], axis=1).astype(np.float32, copy=False)
    return motion, "wxyz"


def _detect_input_format(motion_file: str | Path) -> str:
    with Path(motion_file).open() as f:
        first_line = f.readline().strip()
    if not first_line:
        raise ValueError(f"Input file is empty: {motion_file}")
    first_token = first_line.split(",", 1)[0].strip()
    return "legacy" if _is_number(first_token) else "header"


class SmartMotionLoader(MotionLoader):
    def _load_motion(self):
        input_format = args_cli.input_format
        if input_format == "auto":
            input_format = _detect_input_format(self.motion_file)

        if input_format == "legacy":
            super()._load_motion()
            return

        motion, detected_quat_order = _convert_header_csv_to_motion_array(
            motion_file=self.motion_file,
            frame_range=self.frame_range,
            joint_angle_unit=args_cli.joint_angle_unit,
            translate_scale=args_cli.translate_scale,
        )
        # After conversion we always build [pos + quat(wxyz) + dof]
        self.root_quat_order = detected_quat_order
        self._load_from_array(motion)


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    joint_names: list[str] | None = None,
    MotionLoaderCls=MotionLoader,
):
    """Run one replay and save the resulting motion.npz artifact."""
    motion = MotionLoaderCls(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        root_quat_order=args_cli.root_quat_order,
    )
    output_path = Path(args_cli.output_name) / "motion.npz"
    run_motion_replay(
        sim=sim,
        scene=scene,
        motion_loader=motion,
        output_path=output_path,
        simulation_app=simulation_app,
        joint_names=joint_names or G1_JOINT_NAMES,
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim=sim, scene=scene, joint_names=G1_JOINT_NAMES, MotionLoaderCls=SmartMotionLoader)


if __name__ == "__main__":
    main()
    os._exit(0)  # type: ignore
