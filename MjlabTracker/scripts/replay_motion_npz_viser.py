#!/usr/bin/env python3
"""Replay a G1 motion.npz directly in MuJoCo/Viser.

This is a reference-motion replay tool: it writes the npz root pose and
joint positions into MuJoCo qpos frame by frame. It does not run a policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np
import viser

from mjlab.viewer.viser.scene import MjlabViserScene


DEFAULT_XML = (
    "/home/humanoid/yzh/TextOp/MjlabTracker/source/latent_tracker/"
    "latent_tracker/assets/unitree_description/mjcf/g1_act.xml"
)


def _decode_names(values: np.ndarray) -> list[str]:
    return [str(value.decode("utf-8") if isinstance(value, bytes) else value) for value in values]


def _joint_reindex(model: mujoco.MjModel, motion_joint_names: list[str]) -> list[int]:
    name_to_motion_index = {name: i for i, name in enumerate(motion_joint_names)}
    reindex: list[int] = []
    missing: list[str] = []
    for joint_id in range(1, model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name not in name_to_motion_index:
            missing.append(str(joint_name))
        else:
            reindex.append(name_to_motion_index[joint_name])
    if missing:
        raise ValueError(f"Motion file is missing joints required by XML: {missing}")
    return reindex


def _apply_motion_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: dict[str, np.ndarray],
    joint_reindex: list[int],
    frame: int,
    root_body_index: int,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    data.qpos[:3] = motion["body_pos_w"][frame, root_body_index]
    quat = motion["body_quat_w"][frame, root_body_index].astype(np.float64)
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 0.0:
        quat = quat / quat_norm
    data.qpos[3:7] = quat
    data.qpos[7:] = motion["joint_pos"][frame, joint_reindex]
    mujoco.mj_forward(model, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_npz", help="Path to motion.npz")
    parser.add_argument("--xml", default=DEFAULT_XML, help="MJCF XML path")
    parser.add_argument("--root-body", default="pelvis", help="Body in npz used as floating-base pose")
    parser.add_argument("--host", default="0.0.0.0", help="Viser host")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--fps", type=float, default=None, help="Override replay FPS; defaults to npz fps")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=-1, help="End frame, exclusive; -1 means all")
    parser.add_argument("--loop", action="store_true", help="Loop forever")
    parser.add_argument("--once", action="store_true", help="Replay once and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_path = Path(args.motion_npz).expanduser()
    xml_path = Path(args.xml).expanduser()
    if not motion_path.exists():
        raise FileNotFoundError(motion_path)
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    with np.load(motion_path) as npz:
        motion = {key: npz[key].copy() for key in npz.files}

    required_keys = {"joint_pos", "body_pos_w", "body_quat_w", "joint_names", "body_names"}
    missing_keys = required_keys.difference(motion)
    if missing_keys:
        raise ValueError(f"Motion file missing keys: {sorted(missing_keys)}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_names = _decode_names(motion["joint_names"])
    body_names = _decode_names(motion["body_names"])
    if args.root_body not in body_names:
        raise ValueError(f"root body {args.root_body!r} not found in motion body_names")
    root_body_index = body_names.index(args.root_body)
    joint_reindex = _joint_reindex(model, joint_names)

    num_frames = int(motion["joint_pos"].shape[0])
    start = max(int(args.start), 0)
    end = num_frames if args.end < 0 else min(int(args.end), num_frames)
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, num_frames={num_frames}")

    fps = float(args.fps) if args.fps is not None else float(np.asarray(motion.get("fps", [50])).reshape(-1)[0])
    frame_dt = 1.0 / max(fps * max(args.speed, 1e-6), 1e-6)

    server = viser.ViserServer(host=args.host, port=args.port, label="motion npz replay")
    scene = MjlabViserScene(server, model, num_envs=1)
    scene.create_scene_gui()

    print(f"[INFO] motion: {motion_path}")
    print(f"[INFO] xml: {xml_path}")
    print(f"[INFO] frames: {start}:{end} / {num_frames}, fps={fps}, speed={args.speed}")
    print(f"[INFO] root body: {args.root_body}")
    print(f"[INFO] Open Viser in browser: http://127.0.0.1:{args.port}")

    try:
        while True:
            for frame in range(start, end):
                tic = time.perf_counter()
                _apply_motion_frame(model, data, motion, joint_reindex, frame, root_body_index)
                scene.update_from_mjdata(data)
                elapsed = time.perf_counter() - tic
                time.sleep(max(frame_dt - elapsed, 0.0))
            if args.once or not args.loop:
                if args.once:
                    break
                while True:
                    time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
