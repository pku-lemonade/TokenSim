"""Immutable configuration namespace with a reusable complete structural hash."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EstimateNamespace:
    """Keep configuration equality while hashing once per namespace extension."""

    values: tuple
    digest: int = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "digest", hash(self.values))

    def __hash__(self):
        return self.digest

    def __add__(self, values):
        return type(self)(self.values + values)
