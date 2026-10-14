"""Resident request cohorts executing the production whole-model AF DAG."""

from dataclasses import dataclass, field
from types import SimpleNamespace

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.af.runtime.admission import ResidentKV
from TokenSim.af.trace.recorder import AFTraceRecorder
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
        iteration = self.executor.submit(
            self.worker_ids[owner.physical_worker], requests, phase
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
