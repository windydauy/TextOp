"""Batch convert KimoMotion csv clips into motion.npz artifacts."""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from motion_conversion_common import (
    clip_segments_to_frame_count,
    read_manifest_rows,
    stitch_csv_motion_clips,
    write_segments_csv,
)


parser = argparse.ArgumentParser(description="Batch convert KimoMotion csv clips into motion.npz artifacts.")
parser.add_argument("--manifest", type=str, required=True, help="Path to the KimoMotion manifest csv.")
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root used to resolve manifest csv_path.")
parser.add_argument(
    "--per_clip_output_root",
    type=str,
    required=True,
    help="Directory where per-clip <clip_id>/motion.npz outputs will be written.",
)
parser.add_argument(
    "--stitched_output_root",
    type=str,
    required=True,
    help="Directory where the stitched motion.npz and segments.csv will be written.",
)
parser.add_argument("--buffer_seconds", type=float, default=0.2, help="Hold-buffer duration inserted between clips.")
parser.add_argument("--root_quat_order", type=str, default="wxyz", choices=("xyzw", "wxyz"))
parser.add_argument("--input_fps", type=int, default=50)
parser.add_argument("--output_fps", type=int, default=50)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from motion_replay_common import ArrayMotionLoader, G1_JOINT_NAMES, MotionLoader, ReplayMotionsSceneCfg, run_motion_replay


def convert_single_csv(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    csv_path: Path,
    output_path: Path,
) -> None:
    motion_loader = MotionLoader(
        motion_file=str(csv_path),
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=None,
        root_quat_order=args_cli.root_quat_order,
    )
    run_motion_replay(
        sim=sim,
        scene=scene,
        motion_loader=motion_loader,
        output_path=output_path,
        simulation_app=simulation_app,
        joint_names=G1_JOINT_NAMES,
    )


def main():
    dataset_root = Path(args_cli.dataset_root)
    manifest_rows = read_manifest_rows(args_cli.manifest)
    per_clip_output_root = Path(args_cli.per_clip_output_root)
    stitched_output_root = Path(args_cli.stitched_output_root)
    buffer_frames = int(round(args_cli.buffer_seconds * args_cli.input_fps))

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    print(f"[INFO]: Converting {len(manifest_rows)} KimoMotion clips with buffer_frames={buffer_frames}")

    for index, row in enumerate(manifest_rows, start=1):
        csv_path = dataset_root / row["csv_path"]
        output_path = per_clip_output_root / row["clip_id"] / "motion.npz"
        print(f"[INFO]: [{index}/{len(manifest_rows)}] Converting {row['clip_id']} -> {output_path}")
        convert_single_csv(sim=sim, scene=scene, csv_path=csv_path, output_path=output_path)

    stitched_motion, raw_segments = stitch_csv_motion_clips(
        manifest_rows=manifest_rows,
        dataset_root=dataset_root,
        buffer_frames=buffer_frames,
    )
    stitched_loader = ArrayMotionLoader(
        motion_array=stitched_motion,
        motion_name="kimomotion_stitched",
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        root_quat_order=args_cli.root_quat_order,
    )
    stitched_motion_path = stitched_output_root / "motion.npz"
    print(f"[INFO]: Converting stitched motion -> {stitched_motion_path}")
    run_motion_replay(
        sim=sim,
        scene=scene,
        motion_loader=stitched_loader,
        output_path=stitched_motion_path,
        simulation_app=simulation_app,
        joint_names=G1_JOINT_NAMES,
    )

    output_segments = clip_segments_to_frame_count(raw_segments, frame_count=stitched_loader.output_frames)
    write_segments_csv(output_segments, stitched_output_root / "segments.csv")
    print(f"[INFO]: Wrote stitched segment metadata -> {stitched_output_root / 'segments.csv'}")


if __name__ == "__main__":
    main()
    os._exit(0)  # type: ignore
