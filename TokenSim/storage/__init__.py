"""Public contracts and loading for optional storage media backends."""

from .config import BackendSpec
from .factory import StorageBackendFactory
from .interfaces import IOCost, IORequest, StorageBackend

__all__ = [
    "BackendSpec",
    "IOCost",
    "IORequest",
    "StorageBackend",
    "StorageBackendFactory",
]
