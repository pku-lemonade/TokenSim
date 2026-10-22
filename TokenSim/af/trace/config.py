"""Compatibility configuration for standalone AF experiments."""

from dataclasses import dataclass, fields
from typing import ClassVar
from TokenSim.trace.config import TraceConfig, EVENT_TYPES, TIMING_FIELDS
from TokenSim.af.trace.metrics import AFTraceEvent

EVENT_FIELDS = tuple(field.name for field in fields(AFTraceEvent))


@dataclass(frozen=True)
class AFTraceConfig(TraceConfig):
    mode: str = "full"
    event_fields: ClassVar[tuple[str, ...]] = EVENT_FIELDS
