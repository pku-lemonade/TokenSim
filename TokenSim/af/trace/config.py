"""Independent collection, validation and export choices for AF experiments."""

from dataclasses import asdict, dataclass
from typing import get_args, get_type_hints

from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.errors import ConfigurationError

EVENT_TYPES = get_args(get_type_hints(AFTraceEvent)["event_type"])
TIMING_FIELDS = (
    "iteration_id",
    "block_id",
    "layer_id",
    "phase",
    "event_type",
    "resource_id",
    "ready_at",
    "start_at",
    "end_at",
    "queue_duration",
    "service_duration",
    "request_count",
    "token_count",
    "stage",
    "direction",
    "worker_id",
    "operation",
    "resource_ids",
    "node_id",
    "predecessor_ids",
)


@dataclass(frozen=True)
class AFTraceConfig:
    """Export filters never remove events needed by full invariant validation.

    ``full`` retains readback and exports events, ``validate`` releases each
    completed iteration, and ``off`` skips event construction entirely.
    Iteration and layer identifiers are zero-based; layer -1 is a model boundary.
    """

    mode: str = "full"
    validate: bool = True
    event_types: tuple[str, ...] | None = None
    iterations: tuple[int, ...] | None = None
    layers: tuple[int, ...] | None = None
    detail: str = "full"

    def __post_init__(self):
        if self.mode not in ("full", "validate", "off"):
            raise ConfigurationError("trace mode must be full, validate or off")
        if type(self.validate) is not bool:
            raise ConfigurationError("trace validation must be a boolean")
        if self.mode == "validate" and not self.validate:
            raise ConfigurationError("validate trace mode requires validation")
        if self.detail not in ("full", "timing"):
            raise ConfigurationError("trace detail must be full or timing")
        for name, minimum in (("iterations", 0), ("layers", -1)):
            values = getattr(self, name)
            if values is not None:
                if not isinstance(values, (list, tuple)) or any(
                    type(value) is not int or value < minimum for value in values
                ):
                    raise ConfigurationError(
                        f"trace {name} must be a sequence of integers >= {minimum}"
                    )
                object.__setattr__(self, name, tuple(dict.fromkeys(values)))
        if self.event_types is not None:
            if not isinstance(self.event_types, (list, tuple)) or any(
                value not in EVENT_TYPES for value in self.event_types
            ):
                raise ConfigurationError(
                    "trace event_types must be a sequence drawn from "
                    + ", ".join(EVENT_TYPES)
                )
            object.__setattr__(
                self, "event_types", tuple(dict.fromkeys(self.event_types))
            )
        if self.mode != "full" and (self.filtered or self.detail != "full"):
            raise ConfigurationError("trace export selections require full mode")

    @property
    def enabled(self):
        return self.mode != "off"

    @property
    def filtered(self):
        return any(
            value is not None
            for value in (self.event_types, self.iterations, self.layers)
        )

    def includes(self, event):
        return (
            (self.event_types is None or event.event_type in self.event_types)
            and (self.iterations is None or event.iteration_id in self.iterations)
            and (self.layers is None or event.layer_id in self.layers)
        )

    def describe(self):
        return dict(
            **asdict(self),
            records_events=self.enabled,
            validates_events=self.enabled and self.validate,
            retains_events=self.mode == "full",
            exports_events=self.mode == "full",
            filter_scope="export_only",
            dependency_scope=(
                "may_reference_omitted_events" if self.filtered else "complete"
            ),
        )
