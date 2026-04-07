from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


def read_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    path = Path(manifest_path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_csv_motion_array(
    motion_file: str | Path,
    frame_range: tuple[int, int] | None = None,
) -> np.ndarray:
    motion_path = Path(motion_file)
    load_kwargs: dict[str, Any] = {"delimiter": ",", "ndmin": 2, "dtype": np.float32}
    if frame_range is not None:
        start, end = frame_range
        load_kwargs["skiprows"] = start - 1
        load_kwargs["max_rows"] = end - start + 1
    motion = np.loadtxt(motion_path, **load_kwargs)
    if motion.ndim != 2:
        raise ValueError(f"Expected 2D motion array from {motion_path}, got shape {motion.shape}")
    if motion.shape[1] < 7:
        raise ValueError(f"Expected at least 7 columns in {motion_path}, got shape {motion.shape}")
    return motion.astype(np.float32, copy=False)


def extract_motion_components(
    motion: np.ndarray,
    root_quat_order: str = "xyzw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_pos = motion[:, :3]
    base_rot = motion[:, 3:7]
    if root_quat_order == "xyzw":
        base_rot = base_rot[:, [3, 0, 1, 2]]
    elif root_quat_order == "wxyz":
        pass
    else:
        raise ValueError(f"Unsupported root quaternion order: {root_quat_order}")
    dof = motion[:, 7:]
    return (
        np.ascontiguousarray(base_pos),
        np.ascontiguousarray(base_rot),
        np.ascontiguousarray(dof),
    )


def stitch_csv_motion_clips(
    manifest_rows: list[dict[str, str]],
    dataset_root: str | Path,
    buffer_frames: int,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    dataset_root = Path(dataset_root)
    stitched_chunks: list[np.ndarray] = []
    segments: list[dict[str, int | str]] = []
    current_frame = 0

    for index, row in enumerate(manifest_rows):
        clip_array = load_csv_motion_array(dataset_root / row["csv_path"])
        if clip_array.shape[0] == 0:
            raise ValueError(f"Empty motion clip: {row['csv_path']}")

        start_frame = current_frame
        end_frame = start_frame + clip_array.shape[0] - 1
        stitched_chunks.append(clip_array)
        current_frame = end_frame + 1

        buffer_start_frame = -1
        buffer_end_frame = -1
        is_last_clip = index == len(manifest_rows) - 1
        if buffer_frames > 0 and not is_last_clip:
            buffer_chunk = np.repeat(clip_array[-1:], buffer_frames, axis=0)
            stitched_chunks.append(buffer_chunk)
            buffer_start_frame = current_frame
            buffer_end_frame = current_frame + buffer_frames - 1
            current_frame = buffer_end_frame + 1

        segments.append(
            {
                "clip_id": row["clip_id"],
                "category": row.get("category", ""),
                "prompt": row.get("prompt", ""),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "buffer_start_frame": buffer_start_frame,
                "buffer_end_frame": buffer_end_frame,
            }
        )

    if not stitched_chunks:
        return np.empty((0, 0), dtype=np.float32), segments
    stitched_motion = np.concatenate(stitched_chunks, axis=0).astype(np.float32, copy=False)
    return stitched_motion, segments


def clip_segments_to_frame_count(
    segments: list[dict[str, int | str]],
    frame_count: int,
) -> list[dict[str, int | str]]:
    if frame_count < 0:
        raise ValueError(f"frame_count must be non-negative, got {frame_count}")
    if frame_count == 0:
        return []

    clipped_segments: list[dict[str, int | str]] = []
    max_frame_index = frame_count - 1
    for segment in segments:
        clipped = deepcopy(segment)
        clipped["start_frame"] = min(int(clipped["start_frame"]), max_frame_index)
        clipped["end_frame"] = min(int(clipped["end_frame"]), max_frame_index)

        buffer_start = int(clipped["buffer_start_frame"])
        buffer_end = int(clipped["buffer_end_frame"])
        if buffer_start >= frame_count:
            clipped["buffer_start_frame"] = -1
            clipped["buffer_end_frame"] = -1
        elif buffer_start >= 0:
            clipped["buffer_start_frame"] = buffer_start
            clipped["buffer_end_frame"] = min(buffer_end, max_frame_index)
        clipped_segments.append(clipped)
    return clipped_segments


def write_segments_csv(
    segments: list[dict[str, int | str]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "category",
        "prompt",
        "start_frame",
        "end_frame",
        "buffer_start_frame",
        "buffer_end_frame",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(segments)
