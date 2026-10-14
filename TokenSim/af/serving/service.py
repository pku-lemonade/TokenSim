"""Resident request cohorts executing the production whole-model AF DAG."""

from dataclasses import dataclass, field
from types import SimpleNamespace

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.af.runtime.admission import ResidentKV
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError


@dataclass
class AFSession:
    request_id: int
    dp_rank: int
    owners: dict = field(default_factory=dict)
    generated: int = 0
    cached_context: int = 0


class AFService:
    def __init__(
        self, env, runtime, *, trace=True, seed=42, providers=None, communication=None
    ):
        self.env, self.runtime = env, runtime
        compute, transfer = providers or build_operator_table_af_providers(
            hardware=runtime.hardware,
            model=runtime.model,
            cluster_config=runtime.cluster,
            worker_configs=runtime.workers,
            placement=runtime.placement,
            dtype=runtime.model.activation_dtype,
            fallback="analytical_only",
        )
        self.trace = AFTraceRecorder(trace)
        self.executor = ModelAFExecutor(
            env,
            runtime.cluster,
            runtime.workers,
            compute,
            transfer,
            runtime.model.model_id,
            runtime.model.activation_dtype,
            runtime.model.num_layers,
            runtime.model.hidden_size,
            runtime_plan=runtime.plan,
            model=runtime.model,
            moe_config=runtime.model.moe,
            trace_recorder=self.trace,
            random_seed=seed,
        )
        if communication is not None:
            if runtime.cluster.af.communication_resource_scope != "endpoint":
                raise ConfigurationError(
                    "shared serving communication requires endpoint scope"
                )
            self.executor.resources.endpoint_resources = communication
        self.memory = ResidentKV(runtime.plan)
        self.sessions = {}
        self.owners = {
            (owner.phase, owner.dp_rank): owner for owner in runtime.plan.request_owners
        }
        self.worker_ids = {
            worker.physical_worker_ref.identity: index
            for index, worker in enumerate(runtime.workers)
        }
        self.completed_passes = []

    def admit(
        self, request_id, dp_rank, *, phase="decode", maximum_context, current_context
    ):
        integer(dp_rank, "serving DP rank")
        owner = self.owners.get((phase, dp_rank))
        if owner is None:
            raise ConfigurationError(
                "serving request requires a configured scheduler owner"
            )
        self.memory.admit(
            request_id,
            owner.domain,
            owner.instance,
            maximum_context=maximum_context,
            current_context=current_context,
        )
        session = AFSession(
            request_id,
            dp_rank,
            {owner.domain: owner.instance},
            cached_context=current_context,
        )
        self.sessions[request_id] = session
        return session

    def forward(self, sessions, *, phase="decode", query_tokens=1, context=None):
        integer(query_tokens, "query tokens", minimum=1)
        if not sessions or len({session.dp_rank for session in sessions}) != 1:
            raise ConfigurationError("an AF micro-batch requires one scheduler owner")
        owner = self.owners[(phase, sessions[0].dp_rank)]
        for session in sessions:
            lease = self.memory.leases[session.request_id]
            if (lease.domain, lease.instance) != (owner.domain, owner.instance):
                raise ConfigurationError(
                    "serving owner change requires an explicit KV handoff"
                )
        requests = []
        for session in sessions:
            length = (
                context
                if context is not None
                else session.cached_context + query_tokens
            )
            integer(length, "forward KV context", minimum=query_tokens)
            if length > self.memory.leases[session.request_id].maximum_context:
                raise ConfigurationError("forward exceeds admitted KV reservation")
            requests.append(
                SimpleNamespace(
                    id=session.request_id,
                    is_prefill=phase == "prefill",
                    prefill_len=length,
                    prefill_compute_len=query_tokens,
                    context_len=length,
                    num_computed_tokens=length - query_tokens,
                    cached_prefill_tokens=length - query_tokens,
                    scheduled_query_tokens=query_tokens,
                    scheduled_tokens=query_tokens,
                    generation_idx=session.generated,
                    af_a_owners=session.owners,
                )
            )
        start = self.env.now
        graph_timing = None
        if self.runtime.cluster.af.compute_execution_mode == "cuda_graph":
            resources = self.executor.resources.host_resources_for(
                (owner.physical_worker,)
            )
            host_requests = [resource.request() for _, resource in resources]
            try:
                yield self.env.all_of(host_requests)
                graph_started = self.env.now
                yield self.env.timeout(self.runtime.cluster.af.graph_submit_us / 1e6)
                graph_timing = (
                    graph_started,
                    self.env.now,
                    tuple(key for key, _ in resources),
                )
            finally:
                for (_, resource), request in zip(resources, host_requests):
                    if request.triggered:
                        resource.release(request)
                    else:
                        request.cancel()
        iteration = self.executor.submit(
            self.worker_ids[owner.physical_worker], requests, phase
        )
        if graph_timing is not None and self.trace.enabled:
            submitted, finished, resource_ids = graph_timing
            self.trace.record(
                AFTraceEvent(
                    iteration_id=iteration.iteration_id,
                    block_id=-1,
                    layer_id=-1,
                    phase=phase,
                    event_type="host_submit",
                    operation="cuda_graph_submit",
                    resource_id=resource_ids[0],
                    resource_ids=resource_ids,
                    ready_at=start,
                    start_at=submitted,
                    end_at=finished,
                    queue_duration=submitted - start,
                    service_duration=finished - submitted,
                    request_count=len(sessions),
                    token_count=query_tokens * len(sessions),
                    source_id="assumption:cuda_graph_submit",
                    match_type="analytical",
                    node_id=f"i{iteration.iteration_id}:cuda_graph_submit",
                    details={"timing_scope": "host_submission"},
                )
            )
        yield self.env.process(self.executor.run_iteration(iteration))
        for session, request in zip(sessions, requests):
            session.cached_context = request.context_len
            self.memory.commit(session.request_id, request.context_len)
        self.completed_passes.append(
            {
                "start": start,
                "end": self.env.now,
                "request_ids": tuple(session.request_id for session in sessions),
                "phase": phase,
                "query_tokens": query_tokens,
            }
        )
        return iteration

    def release(self, session):
        self.memory.release(session.request_id)
        del self.sessions[session.request_id]

    def summary(self):
        return {
            "memory": self.memory.summary(),
            "passes": len(self.completed_passes),
            "resource_capacities": self.executor.resources.resource_capacities(),
            "operator_stats": (
                self.executor.compute_provider.stats_dict()
                if hasattr(self.executor.compute_provider, "stats_dict")
                else {}
            ),
        }
