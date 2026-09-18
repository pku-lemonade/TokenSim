from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from TokenSim.errors import ConfigurationError
from TokenSim.hardware._yaml import load_yaml_mapping, require
from TokenSim.hardware.links import LinkCatalog, LinkClass

FABRICS = ("switched", "all_to_all", "ring", "fat_tree", "bus")


@dataclass(frozen=True)
class TopologyLevel:
    """One level of a hierarchical interconnect.

    Level 0 groups devices into the first enclosure (a node or an NVLink
    domain); level ``i`` groups level ``i-1`` groups. ``size`` is the number
    of children per group (``None`` means unbounded, used for the outermost
    level). ``link`` names the :class:`LinkClass` used by an endpoint to talk
    to peers inside this level; ``links_per_endpoint`` is how many such links
    (ports / NICs) each child has, and ``fabric`` describes how they connect.
    """

    name: str
    size: int | None
    link: str
    fabric: str = "switched"
    links_per_endpoint: int = 1
    oversubscription: float = 1.0
    collective_algorithm: str = "auto"

    def __post_init__(self) -> None:
        if self.size is not None and self.size < 1:
            raise ConfigurationError(f"topology level {self.name!r}: size must be >= 1")
        if self.fabric not in FABRICS:
            raise ConfigurationError(
                f"topology level {self.name!r}: fabric must be one of {FABRICS}"
            )
        if self.links_per_endpoint < 1:
            raise ConfigurationError(
                f"topology level {self.name!r}: links_per_endpoint must be >= 1"
            )
        if self.oversubscription < 1.0:
            raise ConfigurationError(
                f"topology level {self.name!r}: oversubscription must be >= 1"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], context: str) -> "TopologyLevel":
        name = str(require(raw, "name", context))
        size = raw.get("size")
        return cls(
            name=name,
            size=int(size) if size is not None else None,
            link=str(require(raw, "link", f"{context} level {name!r}")),
            fabric=str(raw.get("fabric", "switched")),
            links_per_endpoint=int(raw.get("links_per_endpoint", 1)),
            oversubscription=float(raw.get("oversubscription", 1.0)),
            collective_algorithm=str(raw.get("collective_algorithm", "auto")),
        )


@dataclass(frozen=True)
class GroupLayout:
    """How a set of ranks is spread across the topology levels.

    ``fan[i]`` is the largest number of level ``i-1`` groups (devices for
    ``i == 0``) that participate inside one level ``i`` group. A TP group of
    16 GPUs over two 8-GPU nodes has ``fan == (8, 2)``; ``levels`` are the
    topology levels actually crossed (``fan[i] > 1``) plus the innermost.
    """

    ranks: tuple[int, ...]
    fan: tuple[int, ...]
    lowest_common_level: int  # -1: single device, 0: same level-0 group, ...

    @property
    def size(self) -> int:
        return len(self.ranks)

    @property
    def spans_levels(self) -> int:
        return self.lowest_common_level + 1


@dataclass(frozen=True)
class TopologySpec:
    topology_id: str
    levels: tuple[TopologyLevel, ...]
    description: str = ""
    source_id: str = "unspecified"
    # Reserved for measured collective tables keyed by topology.
    collective_table_scope: str | None = None

    def __post_init__(self) -> None:
        if not self.levels:
            raise ConfigurationError(f"topology {self.topology_id!r} needs at least one level")
        for level in self.levels[:-1]:
            if level.size is None:
                raise ConfigurationError(
                    f"topology {self.topology_id!r}: only the outermost level may be unbounded"
                )
        names = [level.name for level in self.levels]
        if len(set(names)) != len(names):
            raise ConfigurationError(f"topology {self.topology_id!r}: level names must be unique")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], context: str = "topology") -> "TopologySpec":
        topology_id = str(require(raw, "topology_id", context))
        if raw.get("schema_version", 1) != 1:
            raise ConfigurationError(f"topology {topology_id!r}: unsupported schema_version")
        levels_raw = require(raw, "levels", f"topology {topology_id!r}")
        if not isinstance(levels_raw, Sequence) or not levels_raw:
            raise ConfigurationError(f"topology {topology_id!r}: levels must be a non-empty list")
        return cls(
            topology_id=topology_id,
            levels=tuple(TopologyLevel.from_mapping(item, f"topology {topology_id!r}") for item in levels_raw),
            description=str(raw.get("description", "")),
            source_id=str(raw.get("source_id", "unspecified")),
            collective_table_scope=raw.get("collective_table_scope"),
        )

    @classmethod
    def two_level(
        cls,
        topology_id: str,
        *,
        node_size: int,
        node_link: str,
        cluster_link: str,
        node_fabric: str = "switched",
        node_links_per_endpoint: int = 1,
        cluster_links_per_endpoint: int = 1,
    ) -> "TopologySpec":
        return cls(
            topology_id=topology_id,
            levels=(
                TopologyLevel(
                    name="node",
                    size=max(1, node_size),
                    link=node_link,
                    fabric=node_fabric,
                    links_per_endpoint=node_links_per_endpoint,
                ),
                TopologyLevel(
                    name="cluster",
                    size=None,
                    link=cluster_link,
                    fabric="fat_tree",
                    links_per_endpoint=cluster_links_per_endpoint,
                ),
            ),
            description="synthesized from cluster networks",
            source_id="derived:cluster-config",
        )

    def validate_links(self, links: LinkCatalog) -> None:
        for level in self.levels:
            if level.link not in links:
                raise ConfigurationError(
                    f"topology {self.topology_id!r} level {level.name!r} references unknown link {level.link!r}"
                )

    # -- coordinates ------------------------------------------------------

    def coordinates(self, device_index: int) -> tuple[int, ...]:
        """Group index of ``device_index`` at every level (innermost first)."""
        if device_index < 0:
            raise ConfigurationError("device index must be non-negative")
        coords: list[int] = []
        cursor = device_index
        for level in self.levels:
            if level.size is None:
                # An unbounded level is a single group containing everything.
                coords.append(0)
                cursor = 0
            else:
                cursor = cursor // level.size
                coords.append(cursor)
        # coords[i] is the level-i group id containing the device.
        return tuple(coords)

    def lowest_common_level(self, device_indices: Sequence[int]) -> int:
        unique = sorted(set(device_indices))
        if len(unique) <= 1:
            return -1
        coords = [self.coordinates(index) for index in unique]
        for level_index in range(len(self.levels)):
            if len({c[level_index] for c in coords}) == 1:
                return level_index
        # Outermost level is unbounded, so this is unreachable; keep a guard.
        return len(self.levels) - 1

    def group_layout(self, device_indices: Sequence[int]) -> GroupLayout:
        ranks = tuple(device_indices)
        unique = sorted(set(ranks))
        if not unique:
            raise ConfigurationError("group layout requires at least one rank")
        common = self.lowest_common_level(unique)
        if common < 0:
            return GroupLayout(ranks=ranks, fan=(1,) * len(self.levels), lowest_common_level=-1)
        coords = [self.coordinates(index) for index in unique]
        fan: list[int] = []
        for level_index in range(len(self.levels)):
            # children at this level: devices (level 0) or level-1 groups
            if level_index == 0:
                children = {(index, c[0]) for index, c in zip(unique, coords)}
                parent_of = {child: child[1] for child in children}
            else:
                children = {tuple(c[level_index - 1 : level_index + 1]) for c in coords}
                parent_of = {child: child[1] for child in children}
            counts: dict[int, int] = {}
            for child, parent in parent_of.items():
                counts[parent] = counts.get(parent, 0) + 1
            fan.append(max(counts.values()))
        return GroupLayout(ranks=ranks, fan=tuple(fan), lowest_common_level=common)

    def level_link(self, level_index: int, links: LinkCatalog) -> LinkClass:
        return links.get(self.levels[level_index].link)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "levels": [
                {
                    "name": level.name,
                    "size": level.size,
                    "link": level.link,
                    "fabric": level.fabric,
                    "links_per_endpoint": level.links_per_endpoint,
                    "oversubscription": level.oversubscription,
                }
                for level in self.levels
            ],
        }


@dataclass(frozen=True)
class TopologyPlacement:
    """Maps simulator worker ids onto device indices of a topology."""

    topology: TopologySpec
    device_index_by_worker: Mapping[int, int] = field(default_factory=dict)

    def device_index(self, worker_id: int) -> int:
        return self.device_index_by_worker.get(worker_id, worker_id)

    def layout(self, worker_ids: Sequence[int]) -> GroupLayout:
        return self.topology.group_layout([self.device_index(w) for w in worker_ids])

    def lowest_common_level(self, worker_ids: Sequence[int]) -> int:
        return self.topology.lowest_common_level([self.device_index(w) for w in worker_ids])

    def level_name(self, level_index: int) -> str:
        if level_index < 0:
            return "local"
        return self.topology.levels[level_index].name


class TopologyCatalog:
    def __init__(self, topologies: Iterable[TopologySpec] = ()) -> None:
        self._topologies: dict[str, TopologySpec] = {}
        for topology in topologies:
            self.add(topology)

    def add(self, topology: TopologySpec) -> None:
        if topology.topology_id in self._topologies:
            raise ConfigurationError(f"duplicate topology_id {topology.topology_id!r}")
        self._topologies[topology.topology_id] = topology

    @classmethod
    def load(cls, directory: str | Path) -> "TopologyCatalog":
        directory = Path(directory)
        if not directory.is_dir():
            raise ConfigurationError(f"topology catalog directory not found: {directory}")
        catalog = cls()
        for path in sorted(directory.glob("*.yaml")):
            document = load_yaml_mapping(path)
            if "topology_id" not in document:
                # links.yaml and other auxiliary files live in the same directory
                continue
            catalog.add(TopologySpec.from_mapping(document, context=str(path)))
        return catalog

    def get(self, topology_id: str) -> TopologySpec:
        try:
            return self._topologies[topology_id]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown topology {topology_id!r}; known: {sorted(self._topologies)}"
            ) from exc

    def __contains__(self, topology_id: object) -> bool:
        return topology_id in self._topologies

    def __iter__(self):
        return iter(self._topologies.values())
