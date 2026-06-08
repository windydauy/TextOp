from __future__ import annotations

import os

import onnx

from mjlab.envs import ManagerBasedRlEnv


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr)


def attach_onnx_metadata(env: ManagerBasedRlEnv, run_path: str, path: str, filename: str = "policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    robot = env.scene["robot"]
    command = env.command_manager.get_term("motion")
    action = env.action_manager.get_term("joint_pos")
    metadata = {
        "run_path": run_path,
        "joint_names": robot.joint_names,
        "default_joint_pos": robot.data.default_joint_pos[0].detach().cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": getattr(action, "_scale", [])[0].detach().cpu().tolist(),
        "anchor_body_name": command.cfg.anchor_body_name,
        "body_names": command.cfg.body_names,
    }

    model = onnx.load(onnx_path)
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = list_to_csv_str(value) if isinstance(value, list) else str(value)
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)
