"""Complete-model prefill on colocated ranks using the ordinary latency backend."""

from types import SimpleNamespace

from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.af.providers.operator_table import analytical_execution_device
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.latency.operator_table.communication import collective_model_for
from TokenSim.llm.llm_request import Request
from TokenSim.parallel import ParallelCommunicator


class _PrefillCommunicator(ParallelCommunicator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.operations = []

    def _record(self, kind, target, bytes_, latency, group, estimate):
        super()._record(kind, target, bytes_, latency, group, estimate)
        self.operations.append((kind, tuple(group), bytes_, latency))

    def estimate_pp_stage_transfer(self, bytes_, count=1):
        if self.rank_info.pp_rank == self.parallel_config.pipeline_parallel_size - 1:
            return 0.0
        return super().estimate_pp_stage_transfer(bytes_, count)


class CompletePrefill:
    def __init__(self, service, seed):
        self.service, self.seed = service, seed
        self.backends = {}
        self.next_id = -1

    def stats_dict(self):
        from TokenSim.latency.operator_table.stats import OperatorStats

        stats = OperatorStats()
        for backend in self.backends.values():
            stats = stats.aggregate(backend.stats)
        return {"fidelity": "prefill_aggregate", **stats.as_dict()}

    def validate(self):
        runtime = self.service.runtime
        if not runtime.cluster.af_link.colocated:
            raise ConfigurationError(
                "prefill requires a complete-model pool with colocated attention and FFN"
            )

    def _backend(self, binding):
        if binding.rank_key in self.backends:
            return self.backends[binding.rank_key]
        service, runtime = self.service, self.service.runtime
        a_domain, f_domain = runtime.cluster.af_domains
        tp = a_domain.parallel.tensor_parallel_size
        ep = f_domain.parallel.expert_parallel_size
        if ep > 1 and ep % tp:
            raise ConfigurationError(
                "complete prefill EP must contain whole TP replicas"
            )
        parallel = ParallelConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=a_domain.parallel.pipeline_parallel_size,
            data_parallel_size=a_domain.parallel.data_parallel_size,
            enable_expert_parallel=ep > 1,
            expert_parallel_scope="global" if ep > tp else "per_dp",
            expert_parallel_size=ep if ep > 1 else None,
        )
        rank = ParallelRankInfo(
            global_rank=binding.domain_rank,
            tp_rank=binding.tp_rank,
            pp_rank=binding.pp_rank,
            dp_rank=binding.dp_rank,
        )
        workers = [
            SimpleNamespace(
                id=service.worker_ids[item.physical_identity],
                dp_rank=item.dp_rank,
                tp_rank=item.tp_rank,
                pp_rank=item.pp_rank,
            )
            for item in runtime.plan.rank_bindings
            if item.domain == a_domain.name
        ]
        device = analytical_execution_device(
            runtime.hardware.device(binding.hardware),
            (
                runtime.cluster.af.graph_device_kernel_us
                if runtime.cluster.af.compute_execution_mode == "cuda_graph"
                else None
            ),
        )
        communicator = _PrefillCommunicator(
            placement=runtime.placement,
            collective_model=collective_model_for(
                device,
                runtime.placement,
                runtime.hardware.links,
                None,
                fallback="analytical_only",
            ),
            workers=workers,
            worker_id=service.worker_ids[binding.physical_identity],
            rank_info=rank,
            parallel_config=parallel,
            dtype=runtime.model.activation_dtype,
        )
        backend = OperatorTableLatencyBackend(
            device=device,
            model=runtime.model,
            parallel_config=parallel,
            rank_info=rank,
            communicator=communicator,
            fallback="analytical_only",
            random_seed=self.seed,
        )
        # One admitted cohort supplies the expert tokens in this forward.
        backend.moe_token_multiplier = 1
        self.backends[binding.rank_key] = backend
        return backend

    def _record(
        self,
        iteration,
        requests,
        operation,
        ready,
        start,
        resources,
        predecessors,
        *,
        event_type="compute",
        end=None,
        **details,
    ):
        service = self.service
        finished = service.env.now if end is None else end
        node = f"prefill:{iteration}:{len(service.trace.events)}"
        service.trace.record(
            AFTraceEvent(
                iteration_id=iteration,
                block_id=-1,
                layer_id=-1,
                phase="prefill",
                event_type=event_type,
                operation=operation,
                resource_id=resources[0],
                resource_ids=tuple(resources),
                ready_at=ready,
                start_at=start,
                end_at=finished,
                queue_duration=start - ready,
                service_duration=finished - start,
                request_count=len(requests),
                token_count=sum(request.step_tokens for request in requests),
                source_id="analytical:complete_model_prefill",
                match_type="analytical",
                source_domain=self.service.runtime.cluster.af_domains[0].name,
                node_id=node,
                predecessor_ids=predecessors,
                details={"fidelity": "prefill_aggregate", **details},
            )
        )
        return (node,)

    def run(self, owner, requests):
        self.validate()
        service, runtime, env = self.service, self.service.runtime, self.service.env
        iteration = self.next_id
        self.next_id -= 1
        complete = []
        for source in requests:
            request = Request(source.id, source.context_len, 1, runtime.plan.block_size)
            request.effective_prefill_tokens = source.prefill_compute_len
            request.num_computed_tokens = source.num_computed_tokens
            request.scheduled_tokens = source.scheduled_query_tokens
            request.scheduled_query_tokens = source.scheduled_query_tokens
            request.emit_logits = source.emit_logits
            complete.append(request)
        bindings = runtime.plan.bindings_for_instance(owner.domain, owner.instance)
        stages = sorted({binding.pp_rank for binding in bindings})
        previous = ()
        started = env.now
        for stage in stages:
            local = [binding for binding in bindings if binding.pp_rank == stage]
            f_domain = runtime.cluster.af_domains[1]
            expert_ranks = (
                [
                    binding
                    for binding in runtime.plan.rank_bindings
                    if binding.domain == owner.domain and binding.pp_rank == stage
                ]
                if f_domain.parallel.expert_parallel_size > len(local)
                else local
            )
            slots = service.executor.resources.compute_resources_for(
                tuple(binding.physical_identity for binding in expert_ranks)
            )
            held = [resource.request() for _, resource in slots]
            ready = env.now
            try:
                yield env.all_of(held)
                start = env.now
                compute, operations = 0.0, {}
                for binding in local:
                    backend = self._backend(binding)
                    backend.communicator.operations.clear()
                    latency = backend.estimate_step_latency(complete)
                    communication = sum(
                        row[3] for row in backend.communicator.operations
                    )
                    compute = max(compute, max(0.0, latency - communication))
                    for kind, group, size, duration in backend.communicator.operations:
                        key = (kind, group, size)
                        operations[key] = max(operations.get(key, 0.0), duration)
                yield env.timeout(compute)
                gpu_ids = tuple(f"worker:{index}" for index, _ in slots)
                previous = self._record(
                    iteration,
                    complete,
                    "complete_model_prefill",
                    ready,
                    start,
                    gpu_ids,
                    previous,
                    logits_rows=(
                        sum(int(request.emit_logits) for request in complete)
                        if stage == stages[-1]
                        else 0
                    ),
                )
            finally:
                for (_, resource), request in zip(slots, held):
                    if request.triggered:
                        resource.release(request)
                    else:
                        request.cancel()
            for (kind, group, size), duration in operations.items():
                endpoints = tuple(
                    runtime.workers[index].physical_worker_ref.identity
                    for index in group
                )
                network = runtime.cluster.af_link.network
                direction = "collective" if kind != "pp_stage_transfer" else "a_to_f"
                registry = service.executor.resources.endpoint_resources
                keys = tuple(
                    registry.format_key(key)
                    for key, _ in registry.for_step(network, direction, endpoints)
                )
                times = yield from registry.run_transfer(
                    keys, duration, 0.0, (iteration, stage, kind, *group)
                )
                previous = self._record(
                    iteration,
                    complete,
                    kind,
                    times[0],
                    times[1],
                    keys,
                    previous,
                    event_type="collective",
                    payload_bytes=size,
                )
        service.finish_trace(iteration)
        return SimpleNamespace(iteration_id=iteration, makespan=env.now - started)
