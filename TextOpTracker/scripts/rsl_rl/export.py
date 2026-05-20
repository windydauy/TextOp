"""Script to export policy.onnx from a checkpoint without launching Isaac Sim UI."""
"""Uses headless mode to minimize resource usage."""

import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
import glob

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Export policy.onnx from a checkpoint.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--resume_path", type=str, default=None, help="Path to the resume checkpoint.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--filename", type=str, default="policy.onnx", help="Output filename for the ONNX model (e.g., policy_XXX.onnx)."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (set to 1 for export).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Force headless mode to avoid launching UI
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app in headless mode
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import textop_tracker.tasks  # noqa: F401
from textop_tracker.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

try:
    from isaaclab.utils.io.pkl import load_pickle
except ImportError:
    load_pickle = None


def load_config(resume_path: str) -> tuple[ManagerBasedRLEnvCfg, RslRlOnPolicyRunnerCfg]:
    """Load the config from the resume path."""
    if load_pickle is None:
        raise ImportError("isaaclab.utils.io.pkl.load_pickle is unavailable in this IsaacLab install")
    param_dir = Path(resume_path).parent / "params"
    env_cfg = load_pickle(str(param_dir / "env.pkl"))
    agent_cfg = load_pickle(str(param_dir / "agent.pkl"))
    return env_cfg, agent_cfg


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Export policy to ONNX format."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # Use minimum number of environments for export
    env_cfg.scene.num_envs = args_cli.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.resume_path:
        assert args_cli.motion_file is not None, "Motion file is required when resume_path is provided"
        resume_path = args_cli.resume_path

        motion_files = glob.glob(str(Path(args_cli.motion_file) / "motion.npz"))
        if not motion_files:
            raise FileNotFoundError(f"No motion.npz found in {Path(args_cli.motion_file)}")
        # Only set motion_files if env_cfg has commands attribute (ManagerBasedRLEnvCfg)
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.commands.motion.motion_files = motion_files  # List[str]

        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        print(f"[INFO]: Using resume path from CLI: {args_cli.resume_path}")
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment (headless mode, minimal setup)
    print("[INFO]: Creating environment in headless mode...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    print("[INFO]: Loading model checkpoint...")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    log_dir = os.path.dirname(resume_path)

    # export policy to onnx
    export_model_dir = os.path.join(log_dir, "exported")
    output_filename = args_cli.filename

    print(f"[INFO]: Exporting policy to ONNX format...")
    print(f"[INFO]: Output directory: {export_model_dir}")
    print(f"[INFO]: Output filename: {output_filename}")

    # Ensure we have a ManagerBasedRLEnv for export
    unwrapped_env = env.unwrapped
    if not isinstance(unwrapped_env, ManagerBasedRLEnv):
        raise TypeError(f"Expected ManagerBasedRLEnv, got {type(unwrapped_env)}")

    export_motion_policy_as_onnx(
        unwrapped_env,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename=output_filename,
    )

    # attach metadata
    run_path = args_cli.wandb_path if args_cli.wandb_path else "none"
    attach_onnx_metadata(unwrapped_env, run_path, export_model_dir, filename=output_filename)

    print(f"[INFO]: Successfully exported policy to: {os.path.join(export_model_dir, output_filename)}")

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
