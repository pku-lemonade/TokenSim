"""Separate native transformer DAGs into stage bodies and outgoing handoffs."""

from collections import defaultdict, deque
from dataclasses import replace
import re

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from scripts.lpx.replay_profiles import NativeProfile
from scripts.lpx.streaming import duration


def split_stages(profile, *, tp, pp):
    """Retain every node once and replace external edges with controller gates.

    A gate waits for the complete preceding part; it can reduce overlap within
    a chunk. A controller may overlap different prompt chunks across stages.
    """
    integer(tp, "stage tensor width", minimum=1)
    integer(pp, "stage pipeline width", minimum=1)
    nodes = []
    for node in profile.nodes:
        if node.release_after is not None:
            raise ConfigurationError(
                "stage splitting does not support cross-part lifetimes"
            )
        if any("/host:" in key for key in node.resources):
            node = replace(
                node, resources=(), primary=None, delay=0, duration=0, occupancy=0
            )
        nodes.append(node)
    children = [[] for _ in nodes]
    incoming = [len(n.predecessors) for n in nodes]
    for i, node in enumerate(nodes):
        for parent in node.predecessors:
            children[parent].append(i)
    ready = deque(i for i, n in enumerate(incoming) if not n)
    owners = {}
    while ready:
        i = ready.popleft()
        node = nodes[i]
        stages = {
            int(match.group(1)) // tp
            for key in node.resources
            if (match := re.search(r"(?:/worker:|lp30:)(\d+)", key))
        }
        previous = max((owners[p] for p in node.predecessors), default=(0, 0))
        if len(stages) == 1:
            owner = (next(iter(stages)), 0)
        elif len(stages) == 2 and max(stages) == min(stages) + 1:
            owner = (min(stages), 1)
        elif not stages and not node.resources:
            owner = previous
        else:
            raise ConfigurationError(
                "stage split has unrecognized or nonlocal resources"
            )
        if owner < previous or owner[0] >= pp:
            raise ConfigurationError(
                "native DAG contains a backward stage/part dependency"
            )
        owners[i] = owner
        for child in children[i]:
            incoming[child] -= 1
            if not incoming[child]:
                ready.append(child)
    if len(owners) != len(nodes):
        raise ConfigurationError("stage split requires an acyclic DAG")
    groups = defaultdict(list)
    for i, owner in owners.items():
        groups[owner].append(i)
    result = {}
    for owner, indices in groups.items():
        indices.sort()
        mapping = {original: new for new, original in enumerate(indices)}
        selected = tuple(
            replace(
                nodes[i],
                predecessors=tuple(
                    mapping[p] for p in nodes[i].predecessors if p in mapping
                ),
            )
            for i in indices
        )
        result[owner] = NativeProfile(
            selected,
            dict(
                modeled_duration_s=duration(selected),
                source_profile=profile.metadata.get("input_sha256"),
                stage=owner[0],
                part=owner[1],
                source_indices=indices,
                scope="Native nodes with complete-part input gates; original host launch is priced at the embedding pool.",
            ),
        )
    if any((stage, 0) not in result for stage in range(pp)) or any(
        (stage, 1) not in result for stage in range(pp - 1)
    ):
        raise ConfigurationError("native split misses a stage body or outgoing handoff")
    return result
