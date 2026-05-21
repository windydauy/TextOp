#!/usr/bin/env python3
"""Visualize an exported ONNX policy tracking a folder of motion.npz files in MuJoCo."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco_viewer
import numpy as np

import deploy_mujoco as dm


def resolve_motion_files(path_arg: str, limit: int = 0, start_index: int = 0) -> list[str]:
    path = Path(path_arg).expanduser()
    if path.is_file():
        files = [path]
    elif any(ch in str(path) for ch in "*?[]"):
        files = sorted(path.parent.glob(path.name))
    else:
        files = sorted(path.glob("**/motion.npz"))
    files = [p for p in files if p.is_file() and p.name == "motion.npz"]
    if start_index > 0:
        files = files[start_index:]
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No motion.npz files found from {path_arg}")
    return [str(p) for p in files]


def make_latent_adapters(task: str, args: argparse.Namespace):
    motion_ae_adapter = None
    motion_transformer_vae_adapter = None
    if task in (dm.TASK_PROJ_GRAV_ANCHOR_OBS_MOTION_AE, dm.TASK_PROJ_GRAV_ANCHOR_EE_OBS):
        motion_ae_adapter = dm.MotionAEOnnxLatentAdapter(
            project_root=args.motion_ae_project_root,
            config_path=args.motion_ae_config_path,
            stats_path=args.motion_ae_stats_path,
            encoder_onnx_path=args.motion_ae_encoder_onnx_path,
            batch_size=args.motion_ae_batch_size,
        )
    elif task in (
        dm.TASK_PROJ_GRAV_ANCHOR_OBS_TRANSFORMER_VAE,
        dm.TASK_PROJ_GRAV_ANCHOR_EE_OBS_TRANSFORMER_VAE,
    ):
        motion_transformer_vae_adapter = dm.MotionTransformerVAEOnnxLatentAdapter(
            project_root=args.motion_transformer_vae_project_root,
            config_path=args.motion_transformer_vae_config_path,
            stats_path=args.motion_transformer_vae_stats_path,
            encoder_onnx_path=args.motion_transformer_vae_encoder_onnx_path,
            batch_size=args.motion_transformer_vae_batch_size,
        )
    return motion_ae_adapter, motion_transformer_vae_adapter


def reset_to_motion_start(data: mujoco.MjData, model: mujoco.MjModel, motion_loader: dm.MotionLoader) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[7:] = motion_loader.joint_pos[0][dm.isaaclab_to_mujoco_reindex]
    data.qpos[:3] = motion_loader.body_pos[0][motion_loader.anchor_body_index]
    data.qpos[3:7] = motion_loader.body_ori[0][motion_loader.anchor_body_index]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def run_one_motion(
    *,
    motion_file: str,
    motion_index: int,
    motion_count: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    session,
    obs_name: str,
    policy_input_dim: int | None,
    task: str,
    observation_terms: tuple[str, ...],
    future_steps: int,
    anchor_body_name: str,
    body_names: list[str],
    motion_ae_adapter,
    motion_transformer_vae_adapter,
    args: argparse.Namespace,
) -> bool:
    print(f"\n[{motion_index}/{motion_count}] visualizing {motion_file}", flush=True)
    motion_loader = dm.MotionLoader(
        motion_file,
        future_steps=future_steps,
        anchor_body_name=anchor_body_name,
        body_names=body_names,
        motion_ae_adapter=motion_ae_adapter,
        motion_transformer_vae_adapter=motion_transformer_vae_adapter,
    )
    reset_to_motion_start(data, model, motion_loader)

    action = np.zeros(dm.NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = motion_loader.joint_pos[0][dm.isaaclab_to_mujoco_reindex].astype(np.float32, copy=True)
    sim_counter = 0
    frame_idx = 0

    while viewer.is_alive and frame_idx < motion_loader.T:
        step_started = time.time()
        tau = dm.pd_control(target_dof_pos, data.qpos[7:], dm.kps, np.zeros_like(dm.kds), data.qvel[6:], dm.kds)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        sim_counter += 1

        if sim_counter % args.control_decimation == 0:
            print(f"motion {motion_index}/{motion_count} frame {frame_idx}/{motion_loader.T}", end="\r", flush=True)
            obs = dm.compute_observation(
                data,
                motion_loader,
                frame_idx,
                action,
                obs_config=args.obs_config,
                task=task,
                observation_terms=observation_terms,
            )
            obs_tensor = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            if policy_input_dim is not None and obs_tensor.shape[1] != policy_input_dim:
                raise RuntimeError(
                    f"Computed obs dim {obs_tensor.shape[1]} does not match ONNX input dim {policy_input_dim}"
                )
            action = session.run(None, {obs_name: obs_tensor})[0].squeeze().astype(np.float32)
            target_dof_pos = action[dm.isaaclab_to_mujoco_reindex] * dm.action_scale + dm.default_angles
            frame_idx += 1

        dm.update_joint_visualization(viewer, motion_loader, min(frame_idx, motion_loader.T - 1))
        viewer.render()

        sleep_time = model.opt.timestep / max(args.realtime_factor, 1e-6) - (time.time() - step_started)
        if sleep_time > 0:
            time.sleep(sleep_time)

    print()
    if viewer.is_alive and args.pause_between_motions > 0:
        time.sleep(args.pause_between_motions)
    return bool(viewer.is_alive)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", required=True, help="Path to exported policy ONNX")
    parser.add_argument("--motion_folder", required=True, help="Folder, glob, or single motion.npz to visualize")
    parser.add_argument("--task", default=dm.TASK_AUTO, choices=(dm.TASK_AUTO, *dm.SUPPORTED_TASKS))
    parser.add_argument("--future_steps", type=int, default=None)
    parser.add_argument("--obs_config", default="ProjGravAnchorEEObs")
    parser.add_argument("--anchor_body_name", default=None)
    parser.add_argument("--limit", type=int, default=2000, help="Visualize only first N motions; 0 means all")
    parser.add_argument("--start_index", type=int, default=0, help="Skip the first N sorted motion files")
    parser.add_argument("--loop", action="store_true", help="Loop over the selected motion list until the viewer closes")
    parser.add_argument("--sim_dt", type=float, default=0.002)
    parser.add_argument("--control_decimation", type=int, default=10)
    parser.add_argument("--realtime_factor", type=float, default=1.0)
    parser.add_argument("--pause_between_motions", type=float, default=0.3)
    parser.add_argument("--camera_distance", type=float, default=3.0)
    parser.add_argument("--camera_azimuth", type=float, default=0.0)
    parser.add_argument("--camera_elevation", type=float, default=-20.0)
    parser.add_argument(
        "--xml_path",
        default="./source/textop_tracker/textop_tracker/assets/unitree_description/mjcf/g1_act.xml",
    )
    parser.add_argument("--motion_ae_encoder_onnx_path", default=dm.DEFAULT_MOTION_AE_ENCODER_ONNX_PATH)
    parser.add_argument("--motion_ae_project_root", default=dm.DEFAULT_MOTION_AE_PROJECT_ROOT)
    parser.add_argument("--motion_ae_config_path", default=dm.DEFAULT_MOTION_AE_CONFIG_PATH)
    parser.add_argument("--motion_ae_stats_path", default=dm.DEFAULT_MOTION_AE_STATS_PATH)
    parser.add_argument("--motion_ae_batch_size", type=int, default=4096)
    parser.add_argument("--motion_transformer_vae_encoder_onnx_path", default=dm.DEFAULT_MOTION_TRANSFORMER_VAE_ENCODER_ONNX_PATH)
    parser.add_argument("--motion_transformer_vae_project_root", default=dm.DEFAULT_MOTION_TRANSFORMER_VAE_PROJECT_ROOT)
    parser.add_argument("--motion_transformer_vae_config_path", default=dm.DEFAULT_MOTION_TRANSFORMER_VAE_CONFIG_PATH)
    parser.add_argument("--motion_transformer_vae_stats_path", default=dm.DEFAULT_MOTION_TRANSFORMER_VAE_STATS_PATH)
    parser.add_argument("--motion_transformer_vae_batch_size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_files = resolve_motion_files(args.motion_folder, limit=args.limit, start_index=args.start_index)

    model = mujoco.MjModel.from_xml_path(str(Path(args.xml_path)))
    model.opt.timestep = args.sim_dt
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.7])
    viewer.cam.distance = args.camera_distance
    viewer.cam.azimuth = args.camera_azimuth
    viewer.cam.elevation = args.camera_elevation

    ort = dm.load_onnxruntime()
    session = ort.InferenceSession(str(Path(args.policy_path).expanduser()))
    obs_name = session.get_inputs()[0].name
    policy_metadata = dm.get_policy_metadata(session)
    policy_input_dim = dm.get_policy_input_dim(session)
    task = dm.resolve_task(
        args.task,
        policy_metadata=policy_metadata,
        obs_config=args.obs_config,
        policy_input_dim=policy_input_dim,
        future_steps=args.future_steps,
    )
    observation_terms = dm.get_observation_terms(task=task)
    future_steps = args.future_steps
    if future_steps is None:
        future_steps = dm.infer_future_steps_from_input_dim(
            policy_input_dim,
            obs_config=args.obs_config,
            task=task,
            observation_terms=observation_terms,
        )
    if future_steps is None:
        raise ValueError("Unable to infer future_steps. Pass --future_steps.")
    anchor_body_name = args.anchor_body_name or policy_metadata.get("anchor_body_name", "pelvis")
    body_names = dm._csv_metadata_list(policy_metadata.get("body_names")) or dm.DEFAULT_BODY_NAMES
    expected_obs_dim = dm.compute_observation_dim(
        future_steps,
        obs_config=args.obs_config,
        task=task,
        observation_terms=observation_terms,
    )
    if policy_input_dim is not None and policy_input_dim != expected_obs_dim:
        raise ValueError(f"Observation dim {expected_obs_dim} does not match ONNX input dim {policy_input_dim}")

    motion_ae_adapter, motion_transformer_vae_adapter = make_latent_adapters(task, args)

    print("visualize task:", task)
    print("visualize observation terms:", ",".join(observation_terms))
    print("visualize future_steps:", future_steps)
    print("visualize anchor_body_name:", anchor_body_name)
    print("visualize num_motions:", len(motion_files))

    try:
        while viewer.is_alive:
            for index, motion_file in enumerate(motion_files, start=1):
                keep_running = run_one_motion(
                    motion_file=motion_file,
                    motion_index=index,
                    motion_count=len(motion_files),
                    model=model,
                    data=data,
                    viewer=viewer,
                    session=session,
                    obs_name=obs_name,
                    policy_input_dim=policy_input_dim,
                    task=task,
                    observation_terms=observation_terms,
                    future_steps=future_steps,
                    anchor_body_name=anchor_body_name,
                    body_names=body_names,
                    motion_ae_adapter=motion_ae_adapter,
                    motion_transformer_vae_adapter=motion_transformer_vae_adapter,
                    args=args,
                )
                if not keep_running:
                    break
            if not args.loop:
                break
    finally:
        if viewer.is_alive:
            viewer.close()


if __name__ == "__main__":
    main()
