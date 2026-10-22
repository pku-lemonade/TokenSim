"""LP30 scoped propagation, synthetic routed port budgets and VXM reductions.

The route generator is an explicit sensitivity topology, not disclosed LPX wiring.
User routes replace generated paths while preserving endpoint and lane budgets.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import math

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class LP30Fabric:
    local_chips: int = 16
    middle_chips: int = 128
    local_us: float = 0.35
    middle_us: float = 0.75
    rack_us: float = 1.15
    ports_per_chip: int = 96
    lanes_per_peer: int = 6
    p2p_lanes: int = 6
    lane_bytes_per_s: float = 14e9
    injection_bytes_per_s: float = 1.25e12
    efficiency: float = 1.0
    reduction_ops_per_s: float = 10e12
    routes: tuple = ()

    def __post_init__(self):
        for name in (
            "local_chips",
            "middle_chips",
            "ports_per_chip",
            "lanes_per_peer",
            "p2p_lanes",
        ):
            integer(getattr(self, name), f"LP30 fabric {name}", minimum=1)
        if self.middle_chips < self.local_chips:
            raise ConfigurationError("LP30 propagation scopes must be nested")
        if max(self.lanes_per_peer, self.p2p_lanes) > self.ports_per_chip:
            raise ConfigurationError("LP30 lane bundle exceeds chip ports")
        for name in ("local_us", "middle_us", "rack_us"):
            number(getattr(self, name), f"LP30 fabric {name}", minimum=0)
        if not self.local_us <= self.middle_us <= self.rack_us:
            raise ConfigurationError("LP30 propagation must increase with scope")
        for name in (
            "lane_bytes_per_s",
            "injection_bytes_per_s",
            "reduction_ops_per_s",
        ):
            number(getattr(self, name), f"LP30 fabric {name}", positive=True)
        number(self.efficiency, "LP30 fabric efficiency", positive=True, maximum=1)
        seen = set()
        for route in self.routes:
            if set(route) != {"source", "target", "via"}:
                raise ConfigurationError("LP30 route requires source, target and via")
            path = (route["source"], *route["via"], route["target"])
            for chip in path:
                integer(chip, "LP30 route chip")
            if len(set(path)) != len(path) or (path[0], path[-1]) in seen:
                raise ConfigurationError("LP30 routes must be unique and acyclic")
            seen.add((path[0], path[-1]))

    def path(self, source, target):
        for route in self.routes:
            if (route["source"], route["target"]) == (source, target):
                return (source, *route["via"], target)
        return (source, target)

    def propagation_s(self, chips):
        if len(set(chips)) <= 1:
            return 0.0
        low, high = min(chips), max(chips)
        if low // self.local_chips == high // self.local_chips:
            return self.local_us * 1e-6
        if low // self.middle_chips == high // self.middle_chips:
            return self.middle_us * 1e-6
        return self.rack_us * 1e-6

    def phase(self, messages, lanes):
        """Return a conservative simultaneous round using real per-port resources.

        Port assignment is deterministic and synthetic; overlapping assignments
        serialize across rounds and incur summed wire loads within a round.
        """
        integer(lanes, "LP30 route lanes", minimum=1)
        if lanes > self.ports_per_chip:
            raise ConfigurationError("LP30 route lane count exceeds physical ports")
        loads, endpoint, lane_loads = Counter(), Counter(), Counter()
        bundle = math.gcd(
            self.ports_per_chip, math.gcd(self.lanes_per_peer, self.p2p_lanes)
        )
        propagation = 0.0
        for source, target, size in messages:
            integer(source, "LP30 source chip")
            integer(target, "LP30 target chip")
            number(size, "LP30 route bytes", minimum=0)
            path = self.path(source, target)
            propagation = max(
                propagation,
                sum(self.propagation_s((a, b)) for a, b in zip(path, path[1:])),
            )
            for a, b in zip(path, path[1:]):
                endpoint[f"lp30:{a}:tx"] += size
                endpoint[f"lp30:{b}:rx"] += size
                for lane in range(lanes):
                    port = (b * lanes + lane) % self.ports_per_chip
                    lane_loads[a, port, "tx"] += size / lanes
                    loads[f"lp30:{a}:port:{port // bundle}:tx"] += size / lanes
                    receive = (a * lanes + lane) % self.ports_per_chip
                    lane_loads[b, receive, "rx"] += size / lanes
                    loads[f"lp30:{b}:port:{receive // bundle}:rx"] += size / lanes
        wire = (
            max(
                max(lane_loads.values(), default=0) / self.lane_bytes_per_s,
                max(endpoint.values(), default=0) / self.injection_bytes_per_s,
            )
            / self.efficiency
        )
        return tuple(sorted(loads.keys() | endpoint.keys())), wire, propagation


class FabricRefinement:
    def __init__(self, builder, policy):
        self.builder, self.policy = builder, policy
        self.counts = Counter()
        self.phase_cache = {}

    def chips(self, endpoints):
        return tuple(self.builder.worker_indices[e] for e in endpoints)

    def vxm_resources(self, endpoints):
        return tuple(f"lp30:{chip}:vxm" for chip in self.chips(endpoints))

    def routed_phase(self, messages, lanes):
        allocated = set(self.builder.worker_indices.values())
        if any(
            chip not in allocated
            for source, target, _ in messages
            for chip in self.policy.path(source, target)
        ):
            raise ConfigurationError(
                "LP30 route uses chips outside the allocated physical inventory"
            )
        return self.policy.phase(messages, lanes)

    def collective_phase(self, endpoints, size):
        chips = self.chips(endpoints)
        key = chips, size
        if key not in self.phase_cache:
            # Each of the two direct phases sends size/N bytes to every peer.
            messages = [
                (a, b, size / len(chips)) for a in chips for b in chips if a != b
            ]
            lanes = min(
                self.policy.lanes_per_peer,
                max(1, self.policy.ports_per_chip // (len(chips) - 1)),
            )
            self.phase_cache[key] = self.routed_phase(messages, lanes)
        return self.phase_cache[key]

    def vector(self, ready, endpoints, size, element_bytes, rank):
        b = self.builder
        resources, wire, propagation = self.collective_phase(endpoints, size)
        for phase in range(2):
            sent = b.append((ready,), resources, duration=wire, rank=rank)
            ready = b.append((sent,), delay=propagation, rank=rank)
            if phase == 0:
                ops = size / element_bytes * (len(endpoints) - 1) / len(endpoints)
                ready = b.append(
                    (ready,),
                    self.vxm_resources(endpoints),
                    duration=ops / self.policy.reduction_ops_per_s,
                    rank=rank,
                )
                self.counts["reduction_operations_per_endpoint"] += ops
        self.counts["vector_rounds"] += 1
        return ready

    def whole_collective(self, index, event, launch_s):
        b, original = self.builder, self.builder.nodes[index]
        endpoints, size = tuple(event.physical_endpoints), event.payload_bytes
        element = 2 if event.details["wire_dtype"] in {"bf16", "fp16"} else 1
        ready = b.append(
            original.predecessors, delay=original.delay + launch_s, rank=original.rank
        )
        if event.operation == "all_reduce":
            ready = self.vector(ready, endpoints, size, element, original.rank)
        else:
            # AllGather's native input is one shard, ReduceScatter's is full.
            size *= len(endpoints) if event.operation == "all_gather" else 1
            resources, wire, propagation = self.collective_phase(endpoints, size)
            ready = b.append((ready,), resources, duration=wire, rank=original.rank)
            ready = b.append((ready,), delay=propagation, rank=original.rank)
            if event.operation == "reduce_scatter":
                ops = size / element * (len(endpoints) - 1) / len(endpoints)
                ready = b.append(
                    (ready,),
                    self.vxm_resources(endpoints),
                    duration=ops / self.policy.reduction_ops_per_s,
                    rank=original.rank,
                )
        b.barrier(index, (ready,))
        self.counts["whole_collectives"] += 1

    def point_to_point(self, index, event):
        if event.operation != "af_pipeline_handoff" or not event.payload_bytes:
            return
        endpoints = tuple(event.physical_endpoints)
        if len(endpoints) != 2 or any(
            e not in self.builder.workers
            or self.builder.workers[e].hardware != "groq_lp30"
            for e in endpoints
        ):
            raise ConfigurationError("LP30 pipeline routing needs two LP30 endpoints")
        b, original = self.builder, self.builder.nodes[index]
        source, target = self.chips(endpoints)
        resources, wire, propagation = self.routed_phase(
            [(source, target, event.payload_bytes)], self.policy.p2p_lanes
        )
        sent = b.append(
            original.predecessors,
            resources,
            delay=original.delay,
            duration=wire,
            rank=original.rank,
        )
        ready = b.append((sent,), delay=propagation, rank=original.rank)
        b.barrier(index, (ready,))
        self.counts["pipeline_handoffs"] += 1

    def report(self):
        return dict(
            config=asdict(self.policy),
            counters=dict(self.counts),
            status="Synthetic per-port routes and scoped propagation; physical LPX wiring and scheduling are unverified.",
        )
