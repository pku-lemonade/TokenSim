"""Validated global collection and export choices."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import ClassVar, get_args, get_type_hints

from TokenSim.trace.events import TraceEvent
from TokenSim.errors import ConfigurationError

EVENT_TYPES = get_args(get_type_hints(TraceEvent)["event_type"])
EVENT_FIELDS = tuple(field.name for field in fields(TraceEvent))
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
class TraceConfig:
    """Export filters never remove events needed by full invariant validation.

    ``full`` retains readback and exports events, ``validate`` releases each
    completed iteration, and ``off`` skips event construction entirely.
    Iteration and layer identifiers are zero-based; layer -1 is a model boundary.
    """

    mode: str = "off"
    output: str | None = None
    format: str = "json"
    workers: tuple[int, ...] | None = None
    phases: tuple[str, ...] | None = None
    event_fields: ClassVar[tuple[str, ...]] = EVENT_FIELDS
    validate: bool = True
    event_types: tuple[str, ...] | None = None
    iterations: tuple[int, ...] | None = None
    layers: tuple[int, ...] | None = None
    detail: str = "full"
    fields: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.output is not None and (
            not isinstance(self.output, str) or not self.output
        ):
            raise ConfigurationError("trace output must be a nonempty path string")
        if self.format not in ("json", "msgpack", "archive"):
            raise ConfigurationError("trace format must be json, msgpack or archive")
        if self.phases is not None:
            if not isinstance(self.phases, (tuple, list)) or any(
                value not in ("prefill", "decode", "mixed") for value in self.phases
            ):
                raise ConfigurationError(
                    "trace phases must be prefill, decode or mixed"
                )
            object.__setattr__(self, "phases", tuple(dict.fromkeys(self.phases)))
        if self.output is not None and self.mode != "full":
            raise ConfigurationError("trace output requires full mode")
        if self.mode not in ("full", "validate", "off"):
            raise ConfigurationError("trace mode must be full, validate or off")
        if type(self.validate) is not bool:
            raise ConfigurationError("trace validation must be a boolean")
        if self.mode == "validate" and not self.validate:
            raise ConfigurationError("validate trace mode requires validation")
        if self.detail not in ("full", "timing"):
            raise ConfigurationError("trace detail must be full or timing")
        for name, minimum in (("iterations", 0), ("layers", -1), ("workers", 0)):
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
        if self.fields is not None:
            if not isinstance(self.fields, (list, tuple)) or any(
                value not in self.event_fields for value in self.fields
            ):
                raise ConfigurationError("trace fields must name TraceEvent fields")
            if self.detail != "full":
                raise ConfigurationError(
                    "trace fields and timing detail are alternatives"
                )
            object.__setattr__(self, "fields", tuple(dict.fromkeys(self.fields)))
        if self.mode != "full" and (
            self.filtered or self.detail != "full" or self.fields is not None
        ):
            raise ConfigurationError("trace export selections require full mode")

    @classmethod
    def from_mapping(cls, value):
        """Read configuration fields only, rejecting unknown or derived settings."""
        names = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) - names:
            raise ConfigurationError(
                "trace configuration must be an object with keys: "
                + ", ".join(sorted(names))
            )
        return cls(**value)

    @property
    def enabled(self):
        return self.mode != "off"

    @property
    def filtered(self):
        return any(
            value is not None
            for value in (
                self.event_types,
                self.iterations,
                self.layers,
                self.workers,
                self.phases,
            )
        )

    def includes(self, event):
        return (
            (self.event_types is None or event.event_type in self.event_types)
            and (self.iterations is None or event.iteration_id in self.iterations)
            and (self.layers is None or event.layer_id in self.layers)
            and (self.workers is None or event.worker_id in self.workers)
            and (self.phases is None or event.phase in self.phases)
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
                "predecessor_ids_omitted"
                if self.fields is not None and "predecessor_ids" not in self.fields
                else "may_reference_omitted_events" if self.filtered else "complete"
            ),
        )
