"""Token blocks."""

from enum import Enum, auto
from typing import Any

from TokenSim.errors import SimulationStateError


class Device(Enum):
    GPU = auto()
    CPU = auto()


class LogicalTokenBlock:
    """A block that stores a contiguous chunk of tokens from left to right.

    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    """

    def __init__(
        self,
        block_id: int,
        block_size: int,
    ) -> None:
        self.block_id = block_id
        self.block_size = block_size

        self.num_tokens = 0

    def is_empty(self) -> bool:
        return self.num_tokens == 0

    def get_num_empty_slots(self) -> int:
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        return self.num_tokens == self.block_size

    def append_tokens(self, num_new_tokens) -> None:
        if num_new_tokens > self.get_num_empty_slots():
            raise SimulationStateError(
                f"cannot append {num_new_tokens} tokens to logical block "
                + f"{self.block_id}; only {self.get_num_empty_slots()} slots left"
            )
        self.num_tokens += num_new_tokens


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
