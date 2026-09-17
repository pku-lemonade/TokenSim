"""Map simulator workers onto topology device indices."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.topology import TopologyPlacement, TopologySpec

logger = logging.getLogger(__name__)


def build_topology_placement(
    cluster_config: Any,
    worker_configs: Sequence[Any],
    hardware: HardwareContext,
) -> TopologyPlacement:
    """Resolve the cluster's topology and each worker's device index.

    When the cluster names a ``topology`` from the catalog, workers are placed
    either at the explicit ``device_indices`` of their group or sequentially.
    Otherwise a two-level topology is synthesized from the legacy ``networks``
    section: workers sharing a named network form one node connected by the
    device's scale-up link, and nodes are joined by the network's link class.
    """
    explicit: dict[int, int] = {}
    cursor = 0
    worker_id = 0
    for group in cluster_config.worker_groups:
        indices = getattr(group, "device_indices", None)
        if indices is not None:
            if len(indices) != group.num_workers:
                raise ConfigurationError(
                    f"worker group {group.role!r}: device_indices has {len(indices)} entries "
                    f"for num_workers={group.num_workers}"
                )
        for position in range(group.num_workers):
            explicit[worker_id] = int(indices[position]) if indices is not None else cursor
            cursor += 1
            worker_id += 1

    topology_id = getattr(cluster_config, "topology", None)
    if topology_id:
        topology = hardware.topology(topology_id)
        if len(set(explicit.values())) != len(explicit):
            raise ConfigurationError("two workers are placed on the same device index")
        return TopologyPlacement(topology=topology, device_index_by_worker=explicit)

    topology = synthesize_topology(cluster_config, worker_configs, hardware)
    # Assign contiguous device indices per named network so that the level-0
    # group boundaries coincide with the network groups.
    node_size = topology.levels[0].size or 1
    per_network: dict[str, int] = {}
    order: list[str] = []
    placement: dict[int, int] = {}
    for index, worker in enumerate(worker_configs):
        network = getattr(worker, "network", "net")
        if network not in per_network:
            per_network[network] = 0
            order.append(network)
        node_index = order.index(network)
        placement[index] = node_index * node_size + per_network[network]
        per_network[network] += 1
    return TopologyPlacement(topology=topology, device_index_by_worker=placement)


def synthesize_topology(
    cluster_config: Any,
    worker_configs: Sequence[Any],
    hardware: HardwareContext,
) -> TopologySpec:
    counts: dict[str, int] = {}
    for worker in worker_configs:
        network = getattr(worker, "network", "net")
        counts[network] = counts.get(network, 0) + 1
    node_size = max(counts.values()) if counts else 1

    first_device = hardware.device(worker_configs[0].hardware) if worker_configs else None
    node_link = None
    if first_device is not None:
        node_link = first_device.scale_up_link or first_device.host_link
    if node_link is None or node_link not in hardware.links:
        node_link = "pcie4_x16"
    node_fabric = "all_to_all" if first_device is not None and first_device.family == "groq_tsp" else "switched"
    node_links = first_device.scale_up_ports if (first_device and first_device.family == "groq_tsp") else 1

    nettypes = {getattr(worker, "nettype", None) for worker in worker_configs}
    nettypes.discard(None)
    cluster_link = None
    for candidate in sorted(nettypes):
        if candidate in hardware.links:
            cluster_link = candidate
            break
    if cluster_link is None:
        cluster_link = "ethernet100Gb"
        if nettypes:
            logger.warning(
                "cluster networks %s are not link classes in the catalog; using %s",
                sorted(nettypes),
                cluster_link,
            )
    if len(nettypes) > 1:
        logger.warning(
            "cluster mixes network link classes %s; the synthesized topology uses %s between nodes",
            sorted(nettypes),
            cluster_link,
        )
    name = "synth_" + "_".join(sorted(counts)) if counts else "synth"
    return TopologySpec.two_level(
        name,
        node_size=node_size,
        node_link=node_link,
        cluster_link=cluster_link,
        node_fabric=node_fabric,
        node_links_per_endpoint=node_links,
    )
