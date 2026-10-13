"""Map simulator workers onto topology device indices."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.topology import TopologyPlacement, TopologySpec
from TokenSim.hardware.domains import CommunicationDomainConfig, physical_domain_config

logger = logging.getLogger(__name__)


def build_topology_placement(
    cluster_config: Any,
    worker_configs: Sequence[Any],
    hardware: HardwareContext,
) -> TopologyPlacement:
    """Resolve the cluster's topology and each worker's device index.

    AF physical mode uses explicit eight-slot communication-domain inventory.
    For other modes, when the cluster names a ``topology`` from the catalog, workers are placed
    either at the explicit ``device_indices`` of their group or sequentially.
    Otherwise a two-level topology is synthesized from the legacy ``networks``
    section: workers sharing a named network form one node connected by the
    device's scale-up link, and nodes are joined by the network's link class.
    """
    af = getattr(cluster_config, "af", None)
    domains = physical_domain_config(af) if af is not None else None
    if domains is not None:
        return build_domain_placement(cluster_config, worker_configs, hardware, domains)
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


def build_domain_placement(cluster_config, worker_configs, hardware, config):
    """Reserve explicit slots first; never expand a declared inventory."""
    config.validate_physical_inventory()
    node_count = config.node_count
    refs = [getattr(worker, "physical_worker_ref", None) for worker in worker_configs]
    if any(ref is None for ref in refs) or len({ref.identity for ref in refs}) != len(refs):
        raise ConfigurationError("AF physical placement needs unique physical worker identities")
    identities = {ref.identity: index for index, ref in enumerate(refs)}
    overrides = dict(config.device_indices)
    for group in cluster_config.worker_groups:
        if group.device_indices is not None:
            for index, slot in enumerate(group.device_indices):
                identity = f"{group.name}:{index}"
                if identity in overrides and overrides[identity] != slot:
                    raise ConfigurationError(f"conflicting device index aliases for {identity}")
                overrides[identity] = slot
    if set(overrides) - set(identities):
        raise ConfigurationError("domain placement override references an unknown worker")
    if len(set(overrides.values())) != len(overrides):
        raise ConfigurationError("domain placement duplicates a GPU slot")
    if any(type(slot) is not int or slot < 0 for slot in overrides.values()):
        raise ConfigurationError("domain placement requires nonnegative integer GPU slots")
    attention = {index for index, worker in enumerate(worker_configs) if "af_attention" in worker.capabilities}
    roles = {index: ("a" if index in attention else "f") for index in range(len(worker_configs))}
    dedicated_nodes = {}
    if config.placement_strategy == "dedicated":
        for identity, slot in overrides.items():
            node = slot // config.node_size
            role = roles[identities[identity]]
            if node in dedicated_nodes and dedicated_nodes[node] != role:
                raise ConfigurationError("dedicated placement override mixes A/F in one node")
            dedicated_nodes[node] = role

    def permitted(worker_id, slot):
        node, local = divmod(slot, config.node_size)
        if node >= node_count or slot < 0:
            return False
        role = roles[worker_id]
        if config.placement_strategy == "mixed":
            return (local < config.mixed_a_slots) == (role == "a")
        if config.placement_strategy == "dedicated":
            return dedicated_nodes.get(node, role) == role
        return True

    placed = {}
    occupied = set(overrides.values())
    for identity, slot in overrides.items():
        worker_id = identities[identity]
        if not permitted(worker_id, slot):
            raise ConfigurationError(f"placement override violates inventory or role policy: {identity}")
        placed[worker_id] = slot
    for worker_id in range(len(worker_configs)):
        if worker_id in placed:
            continue
        slot = next((slot for slot in range(node_count * config.node_size)
                     if slot not in occupied and permitted(worker_id, slot)), None)
        if slot is None:
            raise ConfigurationError("domain placement exceeds declared inventory or role slots")
        placed[worker_id] = slot
        occupied.add(slot)
        if config.placement_strategy == "dedicated":
            dedicated_nodes[slot // config.node_size] = roles[worker_id]
    legacy = synthesize_topology(cluster_config, worker_configs, hardware)
    topology = TopologySpec.two_level("domain_" + legacy.topology_id, node_size=config.node_size,
                                     node_link=legacy.levels[0].link, cluster_link=legacy.levels[-1].link)
    result = TopologyPlacement(topology, placed)
    validate_domain_placement(result, worker_configs, config)
    return result


def validate_domain_placement(placement, worker_configs, config):
    """Validate supplied and generated AF mappings before creating resources."""
    config.validate_physical_inventory()
    if placement is None:
        raise ConfigurationError("AF physical requires placement from explicit communication_domains")
    if placement.topology.levels[0].size != config.node_size:
        raise ConfigurationError("AF physical placement node size must equal communication_domains.node_size=8; migrate legacy placement")
    mapping = placement.device_index_by_worker
    if (set(mapping) != set(range(len(worker_configs)))
            or any(type(rank) is not int for rank in mapping)):
        raise ConfigurationError("AF physical placement must map every actual worker exactly once")
    slots = tuple(mapping.values())
    if any(type(slot) is not int or slot < 0 for slot in slots):
        raise ConfigurationError("AF physical placement requires nonnegative integer GPU slots")
    if len(set(slots)) != len(slots):
        raise ConfigurationError("AF physical placement duplicates a GPU slot")
    if any(slot >= config.node_count * config.node_size for slot in slots):
        raise ConfigurationError("AF physical placement exceeds declared node inventory")
    identities = set()
    node_roles = {}
    for worker_id, worker in enumerate(worker_configs):
        ref = worker.physical_worker_ref
        if ref is None or ref.identity in identities:
            raise ConfigurationError("AF physical placement needs unique physical worker identities")
        identities.add(ref.identity)
        slot = mapping[worker_id]
        for explicit in (config.device_indices.get(ref.identity), worker.device_index):
            if explicit is not None and (type(explicit) is not int or explicit < 0):
                raise ConfigurationError(f"AF placement needs a nonnegative integer GPU slot for {ref.identity}")
            if explicit is not None and explicit != slot:
                raise ConfigurationError(f"AF placement conflicts with explicit GPU slot for {ref.identity}")
        is_a = "af_attention" in worker.capabilities
        node, local = placement.node_id(worker_id), slot % config.node_size
        if config.placement_strategy == "mixed" and (local < config.mixed_a_slots) != is_a:
            raise ConfigurationError("AF mixed placement violates declared role slots")
        if config.placement_strategy == "dedicated":
            if node_roles.setdefault(node, is_a) != is_a:
                raise ConfigurationError("AF dedicated placement mixes A/F in one node")
    if set(config.device_indices) - identities:
        raise ConfigurationError("AF placement override references an unknown physical worker")


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
