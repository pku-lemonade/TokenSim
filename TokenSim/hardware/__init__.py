"""Hardware catalog: device specs, link classes, and hierarchical topologies.

The catalog replaces the monolithic ``TransformerRoofline`` hardware table.
Every numeric field carries a ``source_id`` so that specifications, project
assumptions, and calibrated values can be told apart in results.

``HardwareContext`` is imported lazily because it depends on the model catalog
in ``TokenSim.config``, which in turn uses the dtype helpers defined here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from TokenSim.hardware.device import (
    DeviceCatalog,
    DeviceSpec,
    SourcedValue,
    dtype_bytes,
    normalize_dtype,
)
from TokenSim.hardware.links import LinkCatalog, LinkClass
from TokenSim.hardware.topology import (
    GroupLayout,
    TopologyCatalog,
    TopologyLevel,
    TopologyPlacement,
    TopologySpec,
)

if TYPE_CHECKING:  # pragma: no cover
    from TokenSim.hardware.context import HardwareContext

__all__ = [
    "DeviceCatalog",
    "DeviceSpec",
    "GroupLayout",
    "HardwareContext",
    "LinkCatalog",
    "LinkClass",
    "SourcedValue",
    "TopologyCatalog",
    "TopologyLevel",
    "TopologyPlacement",
    "TopologySpec",
    "dtype_bytes",
    "normalize_dtype",
]


def __getattr__(name: str):
    if name == "HardwareContext":
        from TokenSim.hardware.context import HardwareContext

        return HardwareContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
