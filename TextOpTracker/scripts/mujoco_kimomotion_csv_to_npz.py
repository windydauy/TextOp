"""Convert one KimoMotion numeric csv clip to motion.npz via MuJoCo FK."""

from __future__ import annotations

import argparse

from mujoco_kimomotion_csv_batch_to_npz import convert_single_csv


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert one KimoMotion numeric csv clip to motion.npz via MuJoCo FK."
    )
    parser.add_argument("--input_file", type=str, required=True, help="KimoMotion numeric csv clip to convert.")
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Output .npz path or artifact directory. Directory output becomes <dir>/motion.npz.",
    )
    parser.add_argument(
        "--robot_xml",
        type=str,
        default="TextOpTracker/source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1.xml",
        help="MuJoCo XML path used for forward kinematics.",
    )
    parser.add_argument("--input_fps", type=int, default=50, help="Source KimoMotion csv fps.")
    parser.add_argument("--output_fps", type=int, default=50, help="Target motion.npz fps.")
    parser.add_argument(
        "--root_quat_order",
        type=str,
        default="wxyz",
        choices=("xyzw", "wxyz"),
        help="Quaternion order in KimoMotion root pose columns.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    converted_motion = convert_single_csv(
        csv_path=args.input_file,
        output_name=args.output_name,
        robot_xml=args.robot_xml,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        root_quat_order=args.root_quat_order,
    )
    print(f"[DONE] saved {converted_motion.output_path}")


if __name__ == "__main__":
    main()
