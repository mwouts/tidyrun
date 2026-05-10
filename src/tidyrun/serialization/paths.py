from __future__ import annotations

from os import PathLike
from pathlib import Path

from .types import Location


def to_local_path(location: Location) -> Path:
    if isinstance(location, PathLike):
        return Path(location)

    if "://" in location:
        if location.startswith("file://"):
            return Path(location[len("file://") :])
        raise NotImplementedError(
            "Remote locations are not implemented yet. Use a local path for now."
        )

    return Path(location)


def with_suffix(path: Path, suffix: str) -> Path:
    if path.suffix == suffix:
        return path
    if path.suffix:
        return path.with_suffix(suffix)
    return path.parent / f"{path.name}{suffix}"
