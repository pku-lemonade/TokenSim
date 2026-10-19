"""Bounded F-side batching without changing A ownership or token dependencies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields, replace

import simpy

from TokenSim.errors import ConfigurationError

from TokenSim.af.providers.queries import AFComputeQuery
from TokenSim.af.caching.estimates import PreparedQuery
from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.planning.nodes import ExecutionNode


def _timestamp_boundary(env):
    event = simpy.Event(env)
    event._ok, event._value = True, None
    env.schedule(event, priority=2)
    return event


def merge_queries(members):
    queries = [member.query for member in members]
    first = AFComputeQuery(
        **{
            field.name: getattr(queries[0], field.name)
            for field in fields(AFComputeQuery)
        }
    )
    count = sum(query.local_token_count for query in queries)
    shape = dict(first.local_shape)
    shape["tokens"] = count
    if "elements" in shape:
        shape["elements"] = sum(query.local_shape["elements"] for query in queries)
    workload = Counter()
    for query in queries:
        workload.update(dict(query.expert_workload))
    edges = tuple(edge for query in queries for edge in query.routing_edges)

    def total(name):
        values = [getattr(query, name) for query in queries]
        return sum(values) if all(value is not None for value in values) else None

    return replace(
        first,
        token_count=count,
        local_token_count=count,
        local_shape=shape,
        context_lengths=tuple(c for q in queries for c in q.context_lengths),
        query_lengths=tuple(c for q in queries for c in q.query_lengths),
        expert_ids=tuple(sorted(workload)),
        expert_workload=tuple(sorted(workload.items())),
        routing_edges=edges,
        tensor_slices=tuple(s for q in queries for s in q.tensor_slices),
        original_token_count=(
            len({(e.request_id, e.token_id) for e in edges})
            if edges
            else total("original_token_count")
        ),
        assignment_count=total("assignment_count"),
        input_edge_count=total("input_edge_count"),
        output_token_count=total("output_token_count"),
    )


@dataclass(eq=False)
class BatchMember:
    block: AFBlock
    layer: int
    step: ExecutionNode
    query: AFComputeQuery | PreparedQuery
    done: simpy.Event
    ready: float
    cancelled: bool = False
    group: list[BatchMember] | None = None
    released: simpy.Event | None = None


class FFNBatcher:
    def __init__(self, executor):
        self.executor = executor
        self.env = executor.env
        self.pending = {}
        self.tasks = {}
        self.current = {}
        self.serial = 0
        self.changed = {}

    def eligible(self, step):
        config = self.executor.cluster_config
        if config.af.execution_protocol == "partial_return":
            return config.af.ffn_batching and step.stage in {
                "routed_expert",
                "dense_ffn",
                "shared_expert",
            }
        a_domain, f_domain = config.af_domains
        return (
            a_domain.parallel.data_parallel_size > 1
            and not config.af.communication_timing.native_control_enabled
            and step.source_domain == f_domain.name
            and step.stage in {"router", "routed_expert", "weighted_combine"}
        )

    def run(self, block, layer, step, query, resources, submit_us):
        key = (
            (
                self.executor.native_batches.transaction(block)
                if self.executor.native_batches.enabled
                else None
            ),
            layer,
            step.stage,
            step.physical_workers,
            step.source_domain,
            step.domain_ranks,
            step.tp_group,
            step.ep_group,
            (
                ()
                if self.executor.cluster_config.af.routed_expert_execution == "grouped"
                else query.expert_ids
            ),
            query.phase,
            query.dtype,
            query.weight_dtype,
            query.activation_dtype,
            submit_us,
            tuple(
                (k, v)
                for k, v in sorted(query.local_shape.items())
                if k not in {"tokens", "elements"}
            ),
            step.local_work.source_layout,
            step.local_work.target_layout,
        )
        member = BatchMember(block, layer, step, query, self.env.event(), self.env.now)
        self.pending.setdefault(key, []).append(member)
        changed = self.changed.get(key)
        if changed is not None and not changed.triggered:
            changed.succeed()
        self.changed[key] = self.env.event()
        self.executor._set_active_stage(block, layer, step.node_id, member.ready)
        if key not in self.tasks:
            self.tasks[key] = self.env.process(self._drain(key, resources, submit_us))
        try:
            result, error = yield member.done
            if error is not None:
                raise error
            return result
        finally:
            member.cancelled = True
            self.executor.active_stages.pop(
                (block.iteration_id, block.block_id, step.node_id), None
            )
            if member.group is not None and all(m.cancelled for m in member.group):
                task = self.tasks.get(key)
                if (
                    task is not None
                    and task.is_alive
                    and self.env.active_process is not task
                ):
                    task.interrupt("F batch has no remaining consumers")
                    yield member.released
            elif member.group is None and self.current.get(key) is None:
                if not any(not m.cancelled for m in self.pending.get(key, ())):
                    task = self.tasks.get(key)
                    if task is not None and task.is_alive:
                        task.interrupt("F queue has no remaining consumers")
                        yield task

    def _select(self, key):
        config = self.executor.cluster_config
        if self.executor.native_batches.enabled:
            selected = [m for m in self.pending[key] if not m.cancelled]
            self.pending[key] = []
            return selected
        limit = config.af.ffn_batch_max_tokens or (
            config.af.af_block_size * config.af_domains[0].parallel.data_parallel_size
        )
        selected, retained, owners, tokens = [], [], set(), 0
        for member in self.pending[key]:
            if member.cancelled:
                continue
            owner = (member.block.placement.a_domain, member.block.placement.a_dp_rank)
            count = member.query.original_token_count or member.block.token_count
            if owner in owners or (selected and tokens + count > limit):
                retained.append(member)
            else:
                selected.append(member)
                owners.add(owner)
                tokens += count
        self.pending[key] = retained
        return selected

    def _drain(self, key, resources, submit_us):
        try:
            while any(not m.cancelled for m in self.pending[key]):
                a_domain = self.executor.cluster_config.af_domains[0]
                if self.executor.native_batches.enabled:
                    first_pending = next(
                        m for m in self.pending[key] if not m.cancelled
                    )
                    yield from self.executor.native_batches.wait_turn(
                        first_pending.block,
                        first_pending.layer,
                        first_pending.step.stage,
                    )
                    while (
                        len(
                            {
                                m.block.placement.a_dp_rank
                                for m in self.pending[key]
                                if not m.cancelled
                            }
                        )
                        < self.executor.native_batches.peers
                    ):
                        yield self.changed[key]
                elif a_domain.parallel.data_parallel_size > 1:
                    yield self.env.timeout(
                        self.executor.cluster_config.af.ffn_batch_wait_us / 1e6
                    )
                yield _timestamp_boundary(self.env)
                members = self._select(key)
                if not members:
                    continue
                released = self.env.event()
                self.current[key] = members
                first = members[0]
                collected_at = self.env.now
                collection_started_at = min(m.ready for m in members)
                for member in members:
                    member.group, member.released = members, released
                requests = []
                try:
                    query = merge_queries(members) if len(members) > 1 else first.query
                    config = self.executor.cluster_config
                    if self.executor.native_batches.enabled:
                        limit = (
                            config.af.ffn_batch_max_tokens
                            or config.af.af_block_size
                            * self.executor.native_batches.peers
                        )
                        if sum(m.block.token_count for m in members) > limit:
                            raise ConfigurationError(
                                "native AF peer batch exceeds ffn_batch_max_tokens workspace budget"
                            )
                    submit_id = None
                    if submit_us:
                        submit_id = first.step.node_id + ":batch_submit"
                        yield from self.executor._run_communication_host_stage(
                            first.block,
                            first.layer,
                            first.step,
                            submit_id,
                            first.step.predecessor_ids,
                            first.step.physical_workers,
                            submit_us / 1e6,
                            "host_submit",
                            details={
                                "cost_field": "compute_submit_us",
                                "cost_scope": "per_ffn_batch",
                                "readiness_mode": "invocation_events",
                                "readiness_wait_started_at": collection_started_at,
                            },
                        )
                    device_ready_at = self.env.now
                    requests = [resource.request() for _, resource in resources]
                    yield self.env.all_of(requests)
                    estimate = self.executor._estimate(
                        self.executor.compute_provider,
                        query,
                        first.block,
                        first.layer,
                        first.step.stage,
                    )
                    start = self.env.now
                    for member in members:
                        if member.cancelled:
                            continue
                        self.executor._set_active_stage(
                            member.block,
                            member.layer,
                            member.step.node_id,
                            member.ready,
                            start,
                        )
                    yield self.env.timeout(estimate.service_duration)
                    queue_intervals = (
                        (collection_started_at, collected_at),
                        (device_ready_at, start),
                    )
                    group_id = None
                    if len(members) > 1:
                        group_id = f"{first.step.node_id}:ffn_batch:{self.serial}"
                        self.serial += 1
                        estimate = replace(
                            estimate,
                            details={
                                **estimate.details,
                                "ffn_batch_members": tuple(
                                    m.step.node_id for m in members
                                ),
                                "ffn_batch_original_tokens": query.original_token_count,
                                "ffn_batch_assignments": query.assignment_count,
                                "ffn_batch_expert_workload": query.expert_workload,
                            },
                        )
                    for member in members:
                        if not member.cancelled:
                            member.done.succeed(
                                (
                                    (
                                        (
                                            device_ready_at
                                            if member is first and submit_id
                                            else member.ready
                                        ),
                                        start,
                                        self.env.now,
                                        estimate,
                                        group_id,
                                        submit_id if member is first else None,
                                        member.ready,
                                        queue_intervals,
                                    ),
                                    None,
                                )
                            )
                except simpy.Interrupt:
                    if any(not m.cancelled for m in members):
                        raise
                except Exception as error:
                    for member in members:
                        if not member.cancelled and not member.done.triggered:
                            member.done.succeed((None, error))
                finally:
                    for (_, resource), request in zip(resources, requests):
                        if request.triggered:
                            resource.release(request)
                        else:
                            request.cancel()
                    for member in members:
                        member.group = None
                        self.executor.active_stages.pop(
                            (
                                member.block.iteration_id,
                                member.block.block_id,
                                member.step.node_id,
                            ),
                            None,
                        )
                    self.current.pop(key, None)
                    released.succeed()
        except simpy.Interrupt:
            pass
        except Exception as error:
            for member in self.pending.get(key, ()):
                if not member.cancelled and not member.done.triggered:
                    member.done.succeed((None, error))
        finally:
            self.tasks.pop(key, None)
            self.changed.pop(key, None)
            # An arrival can be enqueued after an empty queue was interrupted
            # and before that cancellation is delivered by SimPy.
            remaining = [
                m
                for m in self.pending.pop(key, ())
                if not m.cancelled and not m.done.triggered
            ]
            if remaining:
                self.pending[key] = remaining
                self.changed[key] = self.env.event()
                self.tasks[key] = self.env.process(
                    self._drain(key, resources, submit_us)
                )
