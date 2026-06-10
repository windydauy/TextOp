# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""DDP training entrypoint for RSL-RL tracker agents.

This script keeps the original train.py untouched and adds a torchrun-friendly
path where each rank loads a static shard of the resolved motion files:
motion_files[rank::world_size].
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager

import torch

# local imports
import cli_args  # isort: skip


def _distributed_context() -> tuple[int, int, int, bool]:
    """Return rank, local_rank, world_size, is_distributed from torchrun env."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, world_size, world_size > 1


@contextmanager
def _file_lock(path: str):
    """Small fcntl-based lock to serialize Isaac app startup across ranks."""
    import fcntl

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


rank, local_rank, world_size, is_distributed = _distributed_context()
rank_device = f"cuda:{local_rank}" if is_distributed else None


def _ddp_debug_enabled() -> bool:
    return os.environ.get("TEXTOP_DDP_DEBUG", "0").lower() not in {"", "0", "false", "no", "off"}


def _ddp_log(message: str, *, always: bool = False) -> None:
    if always or _ddp_debug_enabled():
        print(f"[DDP][rank {rank}/{world_size} local_rank={local_rank}] {message}", flush=True)


def _ddp_backend() -> str:
    return os.environ.get("TEXTOP_DDP_BACKEND", "nccl").lower()


def _ddp_barrier() -> None:
    if _ddp_backend() == "nccl":
        torch.distributed.barrier(device_ids=[local_rank])
    else:
        torch.distributed.barrier()


def _initialize_distributed_before_isaac() -> None:
    """Join the torchrun process group before importing Isaac/Kit."""
    if not is_distributed:
        return

    torch.cuda.set_device(local_rank)
    if torch.distributed.is_initialized():
        return

    backend = _ddp_backend()
    _ddp_log(f"initializing {backend} process group before Isaac import")
    init_kwargs = {"backend": backend, "init_method": "env://"}
    if backend == "nccl":
        init_kwargs["device_id"] = torch.device(rank_device)
    torch.distributed.init_process_group(**init_kwargs)
    _ddp_barrier()
    _ddp_log(f"early {backend} barrier passed")


_initialize_distributed_before_isaac()

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="DDP train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate per rank.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--motion_file", type=str, required=True, help="Path, directory, or glob resolving to motion.npz files.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
_ddp_log(f"parsed args task={args_cli.task} device={args_cli.device} motion_file={args_cli.motion_file}")

if args_cli.video:
    if rank == 0:
        print("[DDP]: --video is ignored by train_ddp.py; training videos are disabled.", flush=True)
    args_cli.video = False
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = False

if is_distributed:
    if args_cli.device not in (None, rank_device):
        print(
            f"[DDP][rank {rank}] Overriding --device={args_cli.device!r} with {rank_device!r} "
            "to match LOCAL_RANK.",
            flush=True,
        )
    args_cli.device = rank_device
    args_cli.distributed = True
    args_cli.multi_gpu = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app; serialize startup to avoid multi-process launcher races
_ddp_log("waiting for Isaac AppLauncher lock")
simulation_app = None
with _file_lock("/tmp/textop_isaaclab_app_launcher.lock"):
    _ddp_log("starting Isaac AppLauncher")
    app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_ddp_log("Isaac AppLauncher ready")
"""Rest everything follows."""

import glob
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

try:
    from isaaclab.utils.io import dump_pickle, dump_yaml
except ImportError:
    import pickle
    import yaml

    def dump_pickle(filename, data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def dump_yaml(filename, data, sort_keys=False):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=sort_keys, allow_unicode=True)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import textop_tracker.tasks  # noqa: F401
from textop_tracker.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def resolve_motion_files(motion_file_arg: str) -> list[str]:
    """Resolve a motion.npz path, glob, or directory tree into sorted files."""
    path = Path(motion_file_arg).expanduser()
    patterns: list[str] = []

    if path.is_file():
        if path.name != "motion.npz":
            raise FileNotFoundError(f"Motion file must be named motion.npz, got {path}")
        return [str(path)]

    if path.is_dir():
        patterns.append(str(path / "**" / "motion.npz"))
    else:
        raw_arg = str(path)
        if any(ch in raw_arg for ch in "*?[]"):
            patterns.append(raw_arg)
            patterns.append(str(path / "motion.npz"))
            patterns.append(str(path / "**" / "motion.npz"))
        else:
            patterns.append(str(path / "motion.npz"))

    motion_files: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            match_path = Path(match)
            if match_path.is_dir():
                match_path = match_path / "motion.npz"
            if match_path.is_file() and match_path.name == "motion.npz":
                resolved = str(match_path)
                if resolved not in seen:
                    seen.add(resolved)
                    motion_files.append(resolved)
    return sorted(motion_files)


def shard_motion_files(motion_files: list[str], rank: int, world_size: int) -> list[str]:
    """Return the static motion shard for one rank."""
    return motion_files[rank::world_size]


def _print_motion_shard_summary(all_motion_files: list[str], rank_motion_files: list[str]) -> None:
    shard_counts = [len(all_motion_files[r::world_size]) for r in range(world_size)]
    if rank == 0:
        print(
            "[DDP]: Resolved "
            f"{len(all_motion_files)} motion files | world_size={world_size} | shard_counts={shard_counts}",
            flush=True,
        )
    preview = rank_motion_files[:2]
    tail = rank_motion_files[-2:] if len(rank_motion_files) > 2 else []
    print(
        f"[DDP][rank {rank}/{world_size} local_rank={local_rank} device={args_cli.device}] "
        f"Using {len(rank_motion_files)} motion files. preview={preview} tail={tail}",
        flush=True,
    )


def _to_plain_value(value):
    """Convert selected config values into YAML/pickle friendly primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain_value(item) for key, item in value.items()}
    return str(value)


def _ddp_env_summary(env_cfg, motion_files: list[str]) -> dict:
    """Return a lightweight env config summary for DDP logs.

    Isaac Lab mutates config objects during environment construction, and a full
    recursive dump can become expensive or unreliable in multi-process runs.
    The DDP entrypoint records the training-critical overrides explicitly.
    """
    motion_cfg = env_cfg.commands.motion
    rewards = {}
    for name, term in getattr(env_cfg.rewards, "__dict__", {}).items():
        if name.startswith("__") or term is None or not hasattr(term, "weight"):
            continue
        reward_entry = {"weight": _to_plain_value(term.weight)}
        params = getattr(term, "params", None)
        if isinstance(params, dict):
            selected_params = {
                key: _to_plain_value(params[key])
                for key in ("pfail_threshold", "threshold", "max_velocity")
                if key in params
            }
            if selected_params:
                reward_entry["params"] = selected_params
        rewards[name] = reward_entry

    return {
        "task": args_cli.task,
        "seed": _to_plain_value(env_cfg.seed),
        "scene": {"num_envs_per_rank": _to_plain_value(env_cfg.scene.num_envs)},
        "sim": {"device": _to_plain_value(env_cfg.sim.device)},
        "ddp": {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "motion_files": _to_plain_value(motion_files),
        },
        "commands": {
            "motion": {
                "anchor_body_name": _to_plain_value(getattr(motion_cfg, "anchor_body_name", None)),
                "body_names": _to_plain_value(getattr(motion_cfg, "body_names", None)),
                "future_steps": _to_plain_value(getattr(motion_cfg, "future_steps", None)),
                "random_static_prob": _to_plain_value(getattr(motion_cfg, "random_static_prob", None)),
                "motion_transformer_vae_enabled": _to_plain_value(
                    getattr(motion_cfg, "motion_transformer_vae_enabled", None)
                ),
                "motion_transformer_vae_project_root": _to_plain_value(
                    getattr(motion_cfg, "motion_transformer_vae_project_root", None)
                ),
                "motion_transformer_vae_config_path": _to_plain_value(
                    getattr(motion_cfg, "motion_transformer_vae_config_path", None)
                ),
                "motion_transformer_vae_checkpoint_path": _to_plain_value(
                    getattr(motion_cfg, "motion_transformer_vae_checkpoint_path", None)
                ),
                "motion_transformer_vae_stats_path": _to_plain_value(
                    getattr(motion_cfg, "motion_transformer_vae_stats_path", None)
                ),
                "motion_transformer_vae_latent_mode": _to_plain_value(
                    getattr(motion_cfg, "motion_transformer_vae_latent_mode", None)
                ),
                "motion_transformer_vae_batch_size": _to_plain_value(
                    getattr(motion_cfg, "motion_transformer_vae_batch_size", None)
                ),
                "enable_adaptive_sampling": _to_plain_value(
                    getattr(motion_cfg, "enable_adaptive_sampling", None)
                ),
                "ads_type": _to_plain_value(getattr(motion_cfg, "ads_type", None)),
                "adaptive_alpha": _to_plain_value(getattr(motion_cfg, "adaptive_alpha", None)),
                "adaptive_beta": _to_plain_value(getattr(motion_cfg, "adaptive_beta", None)),
                "adaptive_uniform_ratio": _to_plain_value(
                    getattr(motion_cfg, "adaptive_uniform_ratio", None)
                ),
                "max_prob_over_uniform": _to_plain_value(
                    getattr(motion_cfg, "max_prob_over_uniform", None)
                ),
            }
        },
        "rewards": rewards,
        "hydra_overrides": _to_plain_value(hydra_args),
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent using torchrun static motion sharding."""
    _ddp_log("main entered")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    if is_distributed:
        env_cfg.seed = (agent_cfg.seed if agent_cfg.seed is not None else 0) + rank
        env_cfg.sim.device = rank_device
        agent_cfg.device = rank_device
    else:
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    _ddp_log(
        f"configured env_device={env_cfg.sim.device} agent_device={agent_cfg.device} "
        f"num_envs={env_cfg.scene.num_envs} max_iterations={agent_cfg.max_iterations}"
    )

    all_motion_files = resolve_motion_files(args_cli.motion_file)
    if not all_motion_files:
        raise FileNotFoundError(f"No motion.npz found in {Path(args_cli.motion_file)}")

    rank_motion_files = shard_motion_files(all_motion_files, rank, world_size)
    if not rank_motion_files:
        raise RuntimeError(
            f"Rank {rank} received an empty motion shard from {len(all_motion_files)} files "
            f"and world_size={world_size}. Reduce nproc_per_node or provide more motion files."
        )
    env_cfg.commands.motion.motion_files = rank_motion_files
    _print_motion_shard_summary(all_motion_files, rank_motion_files)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_root_path}", flush=True)
    log_dir = os.environ.get("TEXTOP_DDP_LOG_STAMP") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if rank == 0:
        _ddp_log(f"dumping config to {log_dir}", always=True)
        env_summary = _ddp_env_summary(env_cfg, rank_motion_files)
        agent_summary = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else agent_cfg
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_summary)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_summary)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_summary)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_summary)
        _ddp_log("finished dumping config", always=True)

    _ddp_log("creating gym environment")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    _ddp_log("gym environment created")

    if isinstance(env.unwrapped, DirectMARLEnv):
        _ddp_log("converting multi-agent env to single-agent env")
        env = multi_agent_to_single_agent(env)

    _ddp_log("wrapping env with RslRlVecEnvWrapper")
    env = RslRlVecEnvWrapper(env)
    _ddp_log("constructing OnPolicyRunner")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=None)
    _ddp_log("OnPolicyRunner constructed")
    runner.add_git_repo_to_log(__file__)
    _ddp_log("registered git repo for logging")

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        if rank == 0:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}", flush=True)
        runner.load(resume_path)

    if is_distributed and torch.distributed.is_initialized():
        _ddp_log("waiting at pre-learn barrier")
        _ddp_barrier()
        _ddp_log("passed pre-learn barrier")

    _ddp_log("starting runner.learn", always=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    _ddp_log("runner.learn completed", always=True)

    _ddp_log("closing env")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        print(f"[DDP][rank {rank}] Unhandled exception:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if simulation_app is not None:
            simulation_app.close()
