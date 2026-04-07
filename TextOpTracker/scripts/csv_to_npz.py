"""Replay a motion from a csv file and export it to motion.npz."""

import argparse
import os
from pathlib import Path

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

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from motion_replay_common import G1_JOINT_NAMES, MotionLoader, ReplayMotionsSceneCfg, run_motion_replay


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
    run_simulator(sim=sim, scene=scene, joint_names=G1_JOINT_NAMES)


if __name__ == "__main__":
    main()
    os._exit(0)  # type: ignore
