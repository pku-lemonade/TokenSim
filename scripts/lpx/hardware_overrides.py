"""Apply explicit additional device-memory allowances without editing catalogs."""

from dataclasses import replace

from TokenSim.config.validation import integer, number
from TokenSim.hardware import DeviceCatalog, LinkCatalog, SourcedValue


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
    return replace(
        hardware,
        devices=DeviceCatalog(devices),
        _packages=dict(hardware._packages),
        _package_index=(
            None
            if hardware._package_index is None
            else {
                name: list(values) for name, values in hardware._package_index.items()
            }
        ),
    )


def communication_hypothesis(hardware, latency_us=None, launch_us=None):
    """Apply explicit LP30 timing assumptions without mutating shared catalogs.

    A catalog latency is paid per modeled hop/round. Mapping a published
    SRAM-to-SRAM endpoint latency to that boundary remains an assumption.
    """
    if latency_us is not None:
        number(latency_us, "C2C latency microseconds")
        hardware = replace(
            hardware,
            links=LinkCatalog(
                (
                    replace(
                        link,
                        latency_us=latency_us,
                        source_id="assumption:lpx-communication-variant",
                        grade="D",
                    )
                    if link.link_id in {"groq_lp30_injection", "groq_lp30_c2c_112g"}
                    else link
                )
                for link in hardware.links
            ),
        )
    if launch_us is not None:
        number(launch_us, "collective launch microseconds")
        hardware = replace(
            hardware,
            devices=DeviceCatalog(
                (
                    replace(
                        device,
                        analytical={
                            **device.analytical,
                            "collective_launch_us": launch_us,
                        },
                    )
                    if device.device_id == "groq_lp30"
                    else device
                )
                for device in hardware.devices
            ),
        )
    return hardware
