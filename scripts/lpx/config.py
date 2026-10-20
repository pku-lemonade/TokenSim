"""Validated inputs for the LPX analytical serving experiment."""

from dataclasses import dataclass
import json
from pathlib import Path

from TokenSim.config.validation import integer, number, boolean
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class Layout:
    a_device: str = "rubin"
    f_device: str = "rubin"
    a_tp: int = 8
    a_cp: int = 1
    a_pp: int = 1
    f_tp: int = 1
    f_ep: int = 8
    f_pp: int = 1
    colocated: bool = True

    def __post_init__(self):
        for name in ("a_tp", "a_cp", "a_pp", "f_tp", "f_ep", "f_pp"):
            integer(getattr(self, name), name, minimum=1)
        boolean(self.colocated, "colocated")
        if self.colocated and (
            self.a_device != self.f_device
            or self.a_cp != 1
            or self.a_pp != self.f_pp
            or self.a_tp != self.f_tp * self.f_ep
        ):
            raise ConfigurationError("colocated A/F requires matching rank inventories")

    @property
    def a_ranks(self):
        return self.a_tp * self.a_cp * self.a_pp

    @property
    def f_ranks(self):
        return self.f_tp * self.f_ep * self.f_pp

    def inventory(self):
        result = {self.a_device: self.a_ranks}
        if not self.colocated:
            result[self.f_device] = result.get(self.f_device, 0) + self.f_ranks
        return result


@dataclass(frozen=True)
class Workload:
    cached_tokens: int = 400000
    new_tokens: int = 4000
    output_tokens: int = 400
    prefill_chunk: int = 256
    seed: int = 42
    draft_tokens: int = 4
    acceptance_probability: float = 0.8

    def __post_init__(self):
        for name in (
            "new_tokens",
            "output_tokens",
            "prefill_chunk",
            "draft_tokens",
        ):
            integer(getattr(self, name), name, minimum=1)
        integer(self.cached_tokens, "cached_tokens")
        integer(self.seed, "seed")
        number(
            self.acceptance_probability, "acceptance_probability", minimum=0, maximum=1
        )

    def maximum_context(self, speculative=False):
        return (
            self.cached_tokens
            + self.new_tokens
            + self.output_tokens
            + (self.draft_tokens if speculative else 0)
        )


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    decode: Layout
    prefill: Layout | None = None
    draft: Layout | None = None
    draft_a_replicas: int = 1

    def __post_init__(self):
        integer(self.draft_a_replicas, "draft attention replicas", minimum=1)
        if self.mode != "draft" and self.draft_a_replicas != 1:
            raise ConfigurationError(
                "draft attention replicas require external drafting"
            )
        if self.mode not in {"rubin", "af", "pd", "draft"}:
            raise ConfigurationError(f"unknown scenario mode {self.mode!r}")
        if (self.mode in {"af", "pd"}) != (self.prefill is not None):
            raise ConfigurationError("AF/PD scenarios require a separate prefill pool")
        if self.prefill is not None and not self.prefill.colocated:
            raise ConfigurationError(
                "LPX prefill requires a complete-model colocated layout"
            )
        if self.draft is not None and not self.draft.colocated:
            raise ConfigurationError(
                "LPX draft prefill requires a complete-model colocated layout"
            )
        if (self.mode == "draft") != (self.draft is not None):
            raise ConfigurationError("external draft scenarios require a draft pool")

    def inventory(self):
        result = {}
        for role, layout in (
            ("target", self.decode),
            ("prefill", self.prefill),
            ("draft", self.draft),
        ):
            if layout:
                replicas = self.draft_a_replicas if role == "draft" else 1
                for device, count in layout.inventory().items():
                    result[device] = result.get(device, 0) + count * replicas
        return result


def load_cases(path):
    raw = json.loads(Path(path).read_text())
    if (
        set(raw) != {"schema_version", "workload", "batches", "cases"}
        or raw["schema_version"] != 1
    ):
        raise ConfigurationError("invalid LPX experiment schema")
    cases = []
    for row in raw["cases"]:
        value = dict(row)
        for name in ("decode", "prefill", "draft"):
            if value.get(name) is not None:
                value[name] = Layout(**value[name])
        cases.append(Case(**value))
    batches = raw["batches"]
    if not batches or not cases or len({case.name for case in cases}) != len(cases):
        raise ConfigurationError("experiment requires batches and uniquely named cases")
    for batch in batches:
        integer(batch, "batch", minimum=1)
    return raw, Workload(**raw["workload"]), cases
