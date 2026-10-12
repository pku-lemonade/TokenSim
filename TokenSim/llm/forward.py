"""Shared forward shapes, capture padding and sequence-parallel ownership."""

from dataclasses import dataclass, field
import math

from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class RankShape:
    tokens: int
    query_width: int | None
    context_build: bool
    requests: int | None = None
    valid_tokens: int | None = None
    valid_requests: int | None = None

    @property
    def routed_tokens(self):
        return self.tokens if self.valid_tokens is None else self.valid_tokens

    def __post_init__(self):
        if self.tokens < 0 or not 0 <= self.routed_tokens <= self.tokens:
            raise ConfigurationError(
                "Forward valid rows must fit the full input tensor"
            )
        if self.valid_requests is not None and (
            self.valid_requests < 0
            or self.requests is not None
            and self.valid_requests > self.requests
        ):
            raise ConfigurationError(
                "Forward valid requests must fit the forward request batch"
            )


@dataclass
class ForwardPass:
    """One synchronized forward snapshot, shared by all rank estimators."""

    ranks: dict[int, RankShape]
    cache: dict = field(default_factory=dict)

    def derived(self, ranks: dict[int, RankShape]) -> "ForwardPass":
        """Reuse an identical derived forward across ranks of this step."""
        identity = ("derived_forward", tuple(ranks.items()))
        result = self.cache.get(identity)
        if result is None:
            result = self.cache[identity] = ForwardPass(ranks)
        return result


def sp_valid_rows(valid, full, tp, rank):
    """Pinned sp_padding_mask: contiguous ceil(full/TP) chunks."""
    chunk = math.ceil(full / tp)
    return max(0, min(chunk, valid - rank * chunk))


def uniform_decode_rows(step, query_width, max_num_seqs):
    """Rows in an eligible synchronized FULL_DECODE_ONLY capture, else None."""
    if not step.ranks or not all(
        not shape.context_build and shape.query_width == query_width
        for shape in step.ranks.values()
    ):
        return None
    upper = max(shape.tokens for shape in step.ranks.values())
    if upper > max_num_seqs * query_width:
        return None
    return query_width * math.ceil(upper / query_width)
