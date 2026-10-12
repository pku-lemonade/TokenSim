"""Token blocks."""

from enum import Enum, auto
from typing import Any


class Device(Enum):
    GPU = auto()
    CPU = auto()


class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_id: int,
        block_size: int,
    ) -> None:
        self.device = device
        self.block_number = block_id
        self.block_size = block_size

        self.ref_count = 0
        self.block_hash: Any | None = None
        self.is_full = False
        self.cached = False
        self.last_accessed = 0

    def clear_cache_metadata(self) -> None:
        self.block_hash = None
        self.is_full = False
        self.cached = False
        self.last_accessed = 0

    def __repr__(self) -> str:
        return (
            f"PhysicalTokenBlock(device={self.device}, "
            f"block_number={self.block_number}, "
            f"ref_count={self.ref_count}, "
            f"cached={self.cached}, "
            f"block_hash={self.block_hash})"
        )
