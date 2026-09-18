from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from TokenSim.config.model_config import ModelCatalog, ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceCatalog, DeviceSpec
from TokenSim.hardware.links import LinkCatalog, LinkClass
from TokenSim.hardware.topology import TopologyCatalog, TopologySpec

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass
class HardwareContext:
    """Everything the simulator needs to know about devices, links, models and data.

    ``operator_packages`` maps ``device_id`` to the loaded operator-table
    packages available for it, keyed by backend name. Packages are loaded
    lazily by :meth:`operator_package` so that a run touching one device does
    not parse tables for every device in the catalog.
    """

    devices: DeviceCatalog
    links: LinkCatalog
    topologies: TopologyCatalog
    models: ModelCatalog
    operator_data_root: Path | None = None
    _packages: dict[tuple[str, str], Any] = field(default_factory=dict, repr=False)
    _package_index: dict[str, list[str]] | None = field(default=None, repr=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, root: str | Path = DEFAULT_DATA_ROOT) -> "HardwareContext":
        root_path = Path(root)
        devices = DeviceCatalog.load(root_path / "devices")
        links = LinkCatalog.load(root_path / "topologies" / "links.yaml")
        topologies = TopologyCatalog.load(root_path / "topologies")
        for topology in topologies:
            topology.validate_links(links)
        models = ModelCatalog.load(root_path / "models")
        operator_root = root_path / "operator_data"
        return cls(
            devices=devices,
            links=links,
            topologies=topologies,
            models=models,
            operator_data_root=operator_root if operator_root.is_dir() else None,
        )

    @classmethod
    def in_memory(
        cls,
        devices: Iterable[DeviceSpec],
        models: Iterable[ModelSpec],
        links: Iterable[LinkClass] = (),
        topologies: Iterable[TopologySpec] = (),
    ) -> "HardwareContext":
        """Build a context from Python objects (tests and notebooks)."""
        return cls(
            devices=DeviceCatalog(devices),
            links=LinkCatalog(links),
            topologies=TopologyCatalog(topologies),
            models=ModelCatalog(models),
            operator_data_root=None,
        )

    # -- lookups ------------------------------------------------------------

    def device(self, name: str) -> DeviceSpec:
        return self.devices.get(name)

    def model(self, name: str) -> ModelSpec:
        return self.models.get(name)

    def link(self, link_id: str) -> LinkClass:
        return self.links.get(link_id)

    def topology(self, topology_id: str) -> TopologySpec:
        return self.topologies.get(topology_id)

    # -- operator packages --------------------------------------------------

    def available_backends(self, device_id: str) -> list[str]:
        if self._package_index is None:
            self._package_index = {}
            if self.operator_data_root is not None:
                for device_dir in sorted(self.operator_data_root.iterdir()):
                    if not device_dir.is_dir():
                        continue
                    backends = [
                        p.name
                        for p in sorted(device_dir.iterdir())
                        if p.is_dir() and (p / "generation_meta.yaml").is_file()
                    ]
                    if backends:
                        self._package_index[device_dir.name] = backends
        return list(self._package_index.get(device_id, []))

    def register_package(self, package: Any) -> None:
        """Attach an in-memory :class:`OperatorDataPackage`."""
        self._packages[(package.device_id, package.backend)] = package
        if self._package_index is None:
            self._package_index = {}
        self._package_index.setdefault(package.device_id, [])
        if package.backend not in self._package_index[package.device_id]:
            self._package_index[package.device_id].append(package.backend)

    def operator_package(self, device_id: str, backend: str | None = None):
        """Return the operator package for a device, or ``None`` when none exists."""
        from TokenSim.operator_data.package import OperatorDataPackage

        backends = self.available_backends(device_id)
        if backend is None:
            if not backends:
                return None
            backend = _preferred_backend(backends)
        elif backend not in backends:
            raise ConfigurationError(
                f"device {device_id!r} has no operator data for backend {backend!r}; available: {backends}"
            )
        key = (device_id, backend)
        if key not in self._packages:
            assert self.operator_data_root is not None
            package = OperatorDataPackage.load(self.operator_data_root / device_id / backend)
            if package.device_id != device_id:
                raise ConfigurationError(
                    f"package at {self.operator_data_root / device_id / backend} declares device_id={package.device_id!r}"
                )
            self._packages[key] = package
            logger.info("loaded operator data %s/%s: %s", device_id, backend, package.summary())
        return self._packages[key]


# Preferred operator-data backend when a cluster does not name one. vLLM first
# (project default, decided 2026-09-18), then the other serving stacks, then
# locally measured and finally analytical packages.
_BACKEND_PRIORITY = ("vllm", "trtllm", "sglang", "measured", "cuda", "groq", "analytical")


def _preferred_backend(backends: list[str]) -> str:
    for candidate in _BACKEND_PRIORITY:
        if candidate in backends:
            return candidate
    return sorted(backends)[0]
