"""Independent collective and service observation oracles."""

from contextlib import contextmanager
from dataclasses import asdict
from unittest.mock import patch

from TokenSim.af.execution.executor import AFExecutor


def execute_ring(phase, inputs, element_bytes=2):
    ranks = phase.participants
    size = phase.logical_input_bytes // element_bytes
    if phase.operation == "all_gather":
        values = {rank: [None] * (size * len(ranks)) for rank in ranks}
        for index, rank in enumerate(ranks):
            values[rank][index * size : (index + 1) * size] = inputs[rank]
    else:
        values = {rank: list(inputs[rank]) for rank in ranks}
    for step, messages in enumerate(phase.rounds):
        sends = []
        for message in messages:
            begin, end = (value // element_bytes for value in message.shard)
            sends.append(
                (message.target, begin, end, values[message.source][begin:end])
            )
        reduce = phase.operation == "reduce_scatter" or (
            phase.operation == "all_reduce" and step < len(ranks) - 1
        )
        for target, begin, end, data in sends:
            if reduce:
                values[target][begin:end] = [
                    left + right for left, right in zip(values[target][begin:end], data)
                ]
            else:
                values[target][begin:end] = data
    if phase.operation == "reduce_scatter":
        chunk = size // len(ranks)
        return {
            rank: values[rank][index * chunk : (index + 1) * chunk]
            for index, rank in enumerate(ranks)
        }
    return values


def execute_hierarchy(plan, inputs, element_bytes=2):
    results = {}
    by_node = {}
    for phase in plan.phases:
        if phase.phase_id.startswith("local_rs:"):
            by_node[int(phase.phase_id.split(":")[1])] = phase.participants
    for phase in plan.phases:
        name = phase.phase_id
        if name == "ring":
            results[name] = execute_ring(phase, inputs, element_bytes)
        elif name.startswith("local_rs:"):
            local_inputs = inputs
            if plan.operation == "reduce_scatter":
                local_inputs = results.get(f"input_layout:{name.split(':')[1]}", inputs)
                destination = int(name.split(":")[2])
                chunk = phase.logical_input_bytes // element_bytes
                local_inputs = {
                    rank: local_inputs[rank][
                        destination * chunk : (destination + 1) * chunk
                    ]
                    for rank in phase.participants
                }
            results[name] = execute_ring(phase, local_inputs, element_bytes)
        elif name.startswith("inter_ar:"):
            local = {
                rank: next(
                    result[rank]
                    for key, result in results.items()
                    if key.startswith("local_rs:") and rank in result
                )
                for rank in phase.participants
            }
            results[name] = execute_ring(phase, local, element_bytes)
        elif name.startswith("inter_ag:"):
            results[name] = execute_ring(
                phase,
                {rank: inputs[rank] for rank in phase.participants},
                element_bytes,
            )
        elif name.startswith("local_ag:"):
            prefix = "inter_ar:" if plan.operation == "all_reduce" else "inter_ag:"
            local = {
                rank: next(
                    result[rank]
                    for key, result in results.items()
                    if key.startswith(prefix) and rank in result
                )
                for rank in phase.participants
            }
            results[name] = execute_ring(phase, local, element_bytes)
        elif name.startswith("inter_rs:"):
            slot = int(name.split(":")[1])
            local = {}
            for index, rank in enumerate(phase.participants):
                pieces = [
                    results[f"local_rs:{index}:{destination}"][rank]
                    for destination in range(len(phase.participants))
                ]
                local[rank] = [item for piece in pieces for item in piece]
            results[name] = execute_ring(phase, local, element_bytes)
        elif name.startswith(("layout:", "input_layout:")):
            source = (
                results[phase.predecessor_ids[0]]
                if phase.predecessor_ids
                else {rank: inputs[rank] for rank in phase.participants}
            )
            width = (
                plan.input_bytes_per_rank
                if name.startswith("layout:")
                else plan.input_bytes_per_rank // len(plan.participants)
            ) // element_bytes
            output = {}
            for rank, data in source.items():
                output[rank] = [
                    value
                    for index in phase.layout_permutation
                    for value in data[index * width : (index + 1) * width]
                ]
            results[name] = output
    referenced = {key for phase in plan.phases for key in phase.predecessor_ids}
    final = {}
    for phase in plan.phases:
        if phase.phase_id not in referenced:
            final.update(results[phase.phase_id])
    return final


@contextmanager
def observe_services():
    """Observe scheduling without adding events, resource requests or timeouts."""
    rows = []
    original = AFExecutor._run_resources_stage

    def observed(self, block, layer_id, stage, resources, estimate_factory):
        result = yield from original(
            self, block, layer_id, stage, resources, estimate_factory
        )
        ready, start, end, estimate = result
        rows.append(
            dict(
                iteration=block.iteration_id,
                block=block.block_id,
                layer=layer_id,
                node=stage,
                ready=ready,
                start=start,
                end=end,
                queue=start - ready,
                estimate=asdict(estimate),
            )
        )
        return result

    with patch.object(AFExecutor, "_run_resources_stage", observed):
        yield rows
