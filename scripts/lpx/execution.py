"""Explicit execution-boundary hypotheses for native LPX experiments.

Compiled execution removes logical host launches, while an independently chosen
instruction floor and one invocation charge keep the unknown costs visible.
These settings are sensitivity assumptions, not LP30 calibration measurements.
"""

from dataclasses import asdict, dataclass, replace

from TokenSim.config.validation import number
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import DeviceCatalog
from TokenSim.hardware.links import LinkCatalog


@dataclass(frozen=True)
class ExecutionAssumptions:
    mode: str = "logical"
    kernel_launch_us: float = 0.2
    instruction_floor_us: float = 0.0
    invocation_us: float = 0.0
    c2c_hop_us: float | None = None

    def __post_init__(self):
        if self.mode not in {"logical", "compiled"}:
            raise ConfigurationError("LPX execution mode must be logical or compiled")
        for name in ("kernel_launch_us", "instruction_floor_us", "invocation_us"):
            number(getattr(self, name), f"LPX {name}", minimum=0)
        if self.c2c_hop_us is not None:
            number(self.c2c_hop_us, "LPX C2C hop latency", minimum=0)
        if self.mode == "logical" and self.instruction_floor_us:
            raise ConfigurationError("instruction floor requires compiled execution")
        if self.mode == "compiled" and self.kernel_launch_us != 0.2:
            raise ConfigurationError(
                "compiled execution uses --instruction-floor-us, not --kernel-launch-us"
            )

    @property
    def effective_operator_startup_us(self):
        return (
            self.kernel_launch_us
            if self.mode == "logical"
            else self.instruction_floor_us
        )

    def report(self):
        return dict(
            **asdict(self),
            effective_operator_startup_us=self.effective_operator_startup_us,
            invocation_scope="One whole captured decode DAG, independent of TP and PP.",
            status="Uncalibrated execution-boundary sensitivity assumptions.",
            retained_costs=[
                "Native per-operator compute and SRAM roofline service.",
                "All data dependencies, resource occupancy and required collectives.",
                "C2C serialization, propagation and separately specified collective startup.",
            ],
        )


def with_execution(hardware, assumptions):
    """Clone experiment catalogs; preserve GPU and heterogeneous-link parameters."""
    devices = DeviceCatalog(
        (
            replace(
                device,
                analytical={
                    **device.analytical,
                    "kernel_launch_us": assumptions.effective_operator_startup_us,
                },
            )
            if device.device_id == "groq_lp30"
            else device
        )
        for device in hardware.devices
    )
    links = LinkCatalog(
        (
            replace(
                link,
                latency_us=assumptions.c2c_hop_us,
                grade="D",
                source_id="experiment-lpx-hop-latency-assumption",
            )
            if assumptions.c2c_hop_us is not None
            and link.link_id in {"groq_lp30_injection", "groq_lp30_c2c_112g"}
            else link
        )
        for link in hardware.links
    )
    return replace(
        hardware, devices=devices, links=links, _packages=dict(hardware._packages)
    )


def add_invocation(profile, invocation_us):
    """Gate all roots on one external invocation without occupying chip resources."""
    number(invocation_us, "LPX invocation_us", minimum=0)
    if not invocation_us:
        return profile
    if not profile.nodes or not any(not n.predecessors for n in profile.nodes):
        raise ConfigurationError("LPX invocation requires a nonempty rooted DAG")
    from scripts.lpx.replay_engine import ReplayNode

    gate = len(profile.nodes)
    nodes = tuple(
        replace(node, predecessors=(gate,)) if not node.predecessors else node
        for node in profile.nodes
    ) + (
        ReplayNode((), (), None, 0, invocation_us * 1e-6, 0, 0),
    )
    base = profile.metadata.get(
        "modeled_duration_s", profile.metadata["native_duration_s"]
    )
    return replace(
        profile,
        nodes=nodes,
        metadata={
            **profile.metadata,
            "invocation_us": invocation_us,
            "invocation_count": 1,
            "modeled_duration_s": base + invocation_us * 1e-6,
        },
    )
