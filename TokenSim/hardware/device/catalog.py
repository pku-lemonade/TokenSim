"""Alias-aware collection of :class:`DeviceSpec`."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from TokenSim.errors import ConfigurationError
from TokenSim.hardware._yaml import load_yaml_mapping
from TokenSim.hardware.device.spec import DeviceSpec


class DeviceCatalog:
    """Alias-aware collection of :class:`DeviceSpec`."""

    def __init__(self, devices: Iterable[DeviceSpec] = ()) -> None:
        self._devices: dict[str, DeviceSpec] = {}
        self._index: dict[str, str] = {}
        for device in devices:
            self.add(device)

    def add(self, device: DeviceSpec) -> None:
        if device.device_id in self._devices:
            raise ConfigurationError(f"duplicate device_id {device.device_id!r}")
        self._devices[device.device_id] = device
        for name in device.names:
            key = _alias_key(name)
            existing = self._index.get(key)
            if existing is not None and existing != device.device_id:
                raise ConfigurationError(
                    f"device alias {name!r} is claimed by both {existing!r} and {device.device_id!r}"
                )
            self._index[key] = device.device_id

    @classmethod
    def load(cls, directory: str | Path) -> "DeviceCatalog":
        directory = Path(directory)
        if not directory.is_dir():
            raise ConfigurationError(f"device catalog directory not found: {directory}")
        catalog = cls()
        for path in sorted(directory.glob("*.yaml")):
            catalog.add(DeviceSpec.from_mapping(load_yaml_mapping(path), context=str(path)))
        if not catalog:
            raise ConfigurationError(f"device catalog {directory} has no *.yaml entries")
        return catalog

    def get(self, name: str) -> DeviceSpec:
        device_id = self._index.get(_alias_key(name))
        if device_id is None:
            raise ConfigurationError(
                f"unknown hardware {name!r}; known devices: {sorted(self._devices)}"
            )
        return self._devices[device_id]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _alias_key(name) in self._index

    def __len__(self) -> int:
        return len(self._devices)

    def __iter__(self):
        return iter(self._devices.values())

    @property
    def device_ids(self) -> list[str]:
        return sorted(self._devices)


def _alias_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())
