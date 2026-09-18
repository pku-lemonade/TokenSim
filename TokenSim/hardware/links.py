from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from TokenSim.errors import ConfigurationError
from TokenSim.hardware._yaml import load_yaml_mapping, positive_number, require


@dataclass(frozen=True)
class LinkClass:
    """A class of physical link between two endpoints.

    ``bandwidth_per_direction_bytes_per_s`` is the *per-endpoint injection*
    bandwidth in one direction. For switched fabrics that aggregate several
    lanes (NVLink through NVSwitch) it is the aggregate a single device can
    push into the fabric; for direct point-to-point links (Groq C2C, PCIe) it
    is one link.
    """

    link_id: str
    display_name: str
    bandwidth_per_direction_bytes_per_s: float
    latency_us: float
    duplex: str = "full"
    source_id: str = "unspecified"
    grade: str = "A"
    # Message size at which a streaming transfer reaches half of the peak
    # bandwidth; models the fixed-cost ramp seen in nccl-tests curves.
    half_bandwidth_bytes: float = 1 << 20
    # Fraction of peak per-direction bandwidth a well-tuned collective reaches
    # at large messages (nccl-tests busbw / link peak). Default from public
    # nccl-tests results; re-fit per link with ``operator_data.cli calibrate``.
    collective_efficiency: float = 0.5
    notes: str = ""

    def __post_init__(self) -> None:
        if self.duplex not in {"full", "half"}:
            raise ConfigurationError(f"link {self.link_id!r}: duplex must be full or half")
        if self.bandwidth_per_direction_bytes_per_s <= 0:
            raise ConfigurationError(f"link {self.link_id!r}: bandwidth must be positive")
        if self.latency_us < 0:
            raise ConfigurationError(f"link {self.link_id!r}: latency must be non-negative")
        if not 0 < self.collective_efficiency <= 1:
            raise ConfigurationError(
                f"link {self.link_id!r}: collective_efficiency must be in (0, 1]"
            )

    def efficiency(self, message_bytes: float) -> float:
        """Fraction of peak bandwidth achievable for a message of this size."""
        if message_bytes <= 0:
            return 1.0
        return message_bytes / (message_bytes + self.half_bandwidth_bytes)

    def transfer_us(self, message_bytes: float, streams: int = 1) -> float:
        """Point-to-point time for one message using ``streams`` parallel links."""
        if message_bytes <= 0:
            return 0.0
        bandwidth = self.bandwidth_per_direction_bytes_per_s * max(1, streams)
        return self.latency_us + message_bytes / (bandwidth * self.efficiency(message_bytes)) * 1e6

    @classmethod
    def from_mapping(cls, link_id: str, raw: Mapping[str, Any]) -> "LinkClass":
        context = f"link {link_id!r}"
        return cls(
            link_id=link_id,
            display_name=str(raw.get("display_name", link_id)),
            bandwidth_per_direction_bytes_per_s=positive_number(
                require(raw, "bandwidth_per_direction_bytes_per_s", context),
                "bandwidth_per_direction_bytes_per_s",
                context,
            ),
            latency_us=float(raw.get("latency_us", 0.0)),
            duplex=str(raw.get("duplex", "full")),
            source_id=str(raw.get("source_id", "unspecified")),
            grade=str(raw.get("grade", "A")).upper(),
            half_bandwidth_bytes=float(raw.get("half_bandwidth_bytes", 1 << 20)),
            collective_efficiency=float(raw.get("collective_efficiency", 0.5)),
            notes=str(raw.get("notes", "")),
        )


class LinkCatalog:
    def __init__(self, links: Iterable[LinkClass] = ()) -> None:
        self._links: dict[str, LinkClass] = {}
        for link in links:
            self.add(link)

    def add(self, link: LinkClass) -> None:
        if link.link_id in self._links:
            raise ConfigurationError(f"duplicate link_id {link.link_id!r}")
        self._links[link.link_id] = link

    @classmethod
    def load(cls, path: str | Path) -> "LinkCatalog":
        document = load_yaml_mapping(path)
        links = document.get("links")
        if not isinstance(links, Mapping) or not links:
            raise ConfigurationError(f"{path}: links must be a non-empty mapping")
        return cls(LinkClass.from_mapping(str(link_id), raw) for link_id, raw in links.items())

    def get(self, link_id: str) -> LinkClass:
        try:
            return self._links[link_id]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown link class {link_id!r}; known links: {sorted(self._links)}"
            ) from exc

    def __contains__(self, link_id: object) -> bool:
        return link_id in self._links

    def __iter__(self):
        return iter(self._links.values())

    def __len__(self) -> int:
        return len(self._links)
