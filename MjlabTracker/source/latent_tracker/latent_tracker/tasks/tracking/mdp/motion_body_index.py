from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def resolve_motion_body_indexes(
    *,
    motion_body_names: Any,
    requested_body_names: Sequence[str],
    fallback_body_indexes: Sequence[int],
    motion_file: str,
) -> list[int]:
    """Resolve tracker body names into indexes for one motion file."""
    if motion_body_names is None:
        return [int(index) for index in fallback_body_indexes]

    names = [str(name) for name in list(motion_body_names)]
    name_to_index = {name: index for index, name in enumerate(names)}
    missing = [name for name in requested_body_names if name not in name_to_index]
    if missing:
        raise KeyError(
            f"Motion file {motion_file} is missing body_names required by tracker: {missing}"
        )
    return [name_to_index[name] for name in requested_body_names]
