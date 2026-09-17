"""Device specifications, dtype handling, and the device catalog.

This subpackage was split from a single ``device.py`` for readability.
All public symbols are re-exported here so that existing imports
``from TokenSim.hardware.device import X`` continue to work unchanged.
"""

from TokenSim.hardware.device.catalog import DeviceCatalog
from TokenSim.hardware.device.dtypes import compute_dtype, dtype_bytes, normalize_dtype
from TokenSim.hardware.device.sourced_value import EVIDENCE_GRADES, SourcedValue
from TokenSim.hardware.device.spec import (
    DEFAULT_ANALYTICAL_PARAMETERS,
    SUPPORTED_FAMILIES,
    DeviceSpec,
)

__all__ = [
    "EVIDENCE_GRADES",
    "DEFAULT_ANALYTICAL_PARAMETERS",
    "SUPPORTED_FAMILIES",
    "DeviceCatalog",
    "DeviceSpec",
    "SourcedValue",
    "compute_dtype",
    "dtype_bytes",
    "normalize_dtype",
]
