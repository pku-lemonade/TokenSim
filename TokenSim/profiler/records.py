"""Normalized device spans with auditable membership and clock identities."""

from dataclasses import asdict, dataclass, field
import re


ALIASES = {
    "qkv_projection": "qkv",
    "output_projection": "output",
    "attention_output": "output",
    "attention_core": "attention",
    "a_rmsnorm": "attention_norm",
    "a_residual_add": "attention_residual",
    "f_rmsnorm": "ffn_norm",
    "dense_gate_up": "ffn_up",
    "gate_up": "ffn_up",
    "dense_down": "ffn_down",
    "down": "ffn_down",
    "dense_act": "activation",
    "swiglu": "activation",
    "gelu": "activation",
    "silu_and_mul": "activation",
    "allreduce": "all_reduce",
    "allgather": "all_gather",
    "reducescatter": "reduce_scatter",
    "routed_experts": "moe_module",
    "routed_expert": "moe_module",
    "v_norm": "value_norm",
}


def canonical(name, aliases=None):
    name = str(name).lower().replace("-", "_").strip()
    return (aliases or {}).get(name, ALIASES.get(name, name))


@dataclass
class Span:
    id: str
    operation: str
    rank: str
    forward: int | None
    layer: int | None
    phase: str | None
    start_us: float
    end_us: float
    category: str
    source: str
    clock: str
    members: tuple[str, ...] = ()
    kernel_sum_us: float = 0
    busy_us: float = 0
    shape: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    @property
    def duration_us(self):
        return self.end_us - self.start_us

    def as_dict(self):
        return dict(asdict(self), duration_us=self.duration_us)


def union_duration(intervals):
    end = None
    total = 0.0
    for left, right in sorted(intervals):
        total += max(0.0, right - max(left, end if end is not None else left))
        end = max(right, end if end is not None else right)
    return total


def intervals_for(spans, categories=None):
    intervals = []
    for span in spans:
        details = span.details
        if categories is None:
            intervals.extend(
                details.get("service_intervals", [(span.start_us, span.end_us)])
            )
        elif "compute_intervals" in details:
            if categories & {
                "compute",
                "kernel",
                "gpu_kernel",
                "gpu_memset",
                "control_device",
            }:
                intervals.extend(details["compute_intervals"])
            if categories & {"collective", "transfer", "gpu_memcpy"}:
                intervals.extend(details.get("communication_intervals", ()))
        elif span.category in categories:
            intervals.extend(
                details.get("service_intervals", [(span.start_us, span.end_us)])
            )
    return intervals


def annotation(name):
    match = re.fullmatch(r"tokensim\.layer\.(\d+)(?:\.(.+))?", name)
    if match:
        return int(match[1]), match[2]
    if name.startswith("tokensim.model."):
        return -1, name.removeprefix("tokensim.model.")
    return None
