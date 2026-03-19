"""Version information."""

from importlib import metadata

__all__ = ("__version__",)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available (e.g., development mode)
    __version__ = "0.2.0"
