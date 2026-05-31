from cloudpathlib import CloudPath
from pathlib import Path


def with_suffix(path: Path | CloudPath, suffix: str) -> Path | CloudPath:
    """
    The serialize and deserialize functions always take paths without suffixes.

    This functions adds the suffix to locate the actual file on disk.
    """
    return path.with_name(path.name + suffix)
