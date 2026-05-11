from importlib.metadata import version
from .serialization import LazyDict, deserialize, serialize

__version__ = version("tidyrun")

__all__ = [
    "__version__",
    "LazyDict",
    "deserialize",
    "serialize",
]
