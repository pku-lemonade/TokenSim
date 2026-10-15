"""Apply explicit additional device-memory allowances without editing catalogs."""

from dataclasses import replace

from TokenSim.config.validation import integer
from TokenSim.hardware import DeviceCatalog, SourcedValue


def reserve_memory(hardware, overrides):
    amounts = {}
    for name, value in overrides.items():
        integer(value, f"{name} additional memory reserve")
        amounts[hardware.device(name).device_id] = value
    if not any(amounts.values()):
        return hardware
    devices = []
    for device in hardware.devices:
        extra = amounts.get(device.device_id, 0)
        devices.append(
            replace(
                device,
                memory_reserved_bytes=SourcedValue(
                    device.memory_reserved_bytes.value + extra,
                    "assumption:user-additional-memory-reserve",
                    "D",
                ),
            )
            if extra
            else device
        )
    return replace(hardware, devices=DeviceCatalog(devices), _packages={})
