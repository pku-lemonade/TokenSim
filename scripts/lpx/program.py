"""Compose explicit LP30 operator programs on the native replay resources."""

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace

from TokenSim.errors import ConfigurationError
from scripts.lpx.fabric import FabricRefinement, LP30Fabric
from scripts.lpx.streaming import StreamingBuilder, duration
from TokenSim.operator_data.analytical import analytical_model_for


class LP30Program:
    append = StreamingBuilder.append

    def __init__(self, hardware, chips):
        self.nodes = []
        self.device = hardware.device("groq_lp30")
        if any(
            name not in self.device.analytical
            for name in ("lpx_fabric", "lpx_pipeline")
        ):
            raise ConfigurationError(
                "LP30 program composition needs explicit fabric and pipeline fidelity"
            )
        self.estimator = analytical_model_for(self.device.family)
        self.worker_indices = {f"chip:{chip}": chip for chip in range(chips)}
        self.fabric = FabricRefinement(
            self, LP30Fabric(**self.device.analytical["lpx_fabric"])
        )
        self.operators = Counter()

    def compute(self, parents, chip, table, key):
        estimate = self.estimator.estimate(table, key, self.device)
        resources = (f"mfu-search/worker:{chip}", f"lp30:{chip}:vxm")
        self.operators[table] += 1
        return self.append(parents, resources, duration=estimate.latency_us * 1e-6)

    def transfer(self, parents, messages):
        resources, wire, propagation = self.fabric.routed_phase(
            messages, self.fabric.policy.p2p_lanes
        )
        sent = self.append(parents, resources, duration=wire)
        return self.append((sent,), delay=propagation)

    def reduce(self, parents, chips, size, element_bytes=2):
        if not chips:
            raise ConfigurationError("LP30 program collective needs a participant")
        ready = self.append(parents)
        if len(chips) == 1:
            return ready
        return self.fabric.vector(
            ready, tuple(f"chip:{chip}" for chip in chips), size, element_bytes, 0
        )

    def acquire(self, parents, arena):
        return self.append(parents, (arena,))

    def release_at(self, acquisition, completed):
        self.nodes[acquisition] = replace(
            self.nodes[acquisition], release_after=completed
        )

    def include(self, profile, parents, *, suppress_hosts=False):
        offset = len(self.nodes)
        used = set()
        for node in profile.nodes:
            used.update(node.predecessors)
            shifted = replace(
                node,
                predecessors=(
                    tuple(offset + p for p in node.predecessors)
                    if node.predecessors
                    else tuple(parents)
                ),
                release_after=(
                    offset + node.release_after
                    if node.release_after is not None
                    else None
                ),
            )
            if suppress_hosts and any("/host:" in key for key in node.resources):
                shifted = replace(
                    shifted,
                    resources=(),
                    primary=None,
                    delay=0,
                    duration=0,
                    occupancy=0,
                )
            self.nodes.append(shifted)
        return self.append(
            tuple(offset + i for i in range(len(profile.nodes)) if i not in used)
        )

    def profile(self):
        encoded = json.dumps([asdict(node) for node in self.nodes], sort_keys=True)
        return SimpleNamespace(
            nodes=tuple(self.nodes),
            metadata=dict(
                program_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                modeled_duration_s=duration(self.nodes),
                operators=dict(self.operators),
                scope="Explicit synthetic operator composition with analytical LP30 costs; compiler and physical bank placement unverified.",
            ),
        )
