"""AF execution plans, physical workers and phase selection for the engine."""

from __future__ import annotations

from typing import Any
from TokenSim.af import (
    AFCacheConfig,
    AFComputeLatencyProvider,
    AFTransferLatencyProvider,
    AFExecutor,
    AFIteration,
    AFResourceRegistry,
    AFRuntimePlan,
    compile_af_runtime_plan,
)
from TokenSim.af.model_executor import ModelAFExecutor
from TokenSim.block.replica_kv import ReplicaKVBlockManager
from TokenSim.config.config import WorkerConfig
from TokenSim.errors import ConfigurationError
from TokenSim.llm.llm_request import Request
from .worker import LLMWorker
from .af_worker import AFComputeWorker


class AFEngineMixin:
    def _initialize_workers(
        self,
        kv_transfer_config,
        worker_configs,
        af_compute_provider,
        af_transfer_provider,
        af_dtype,
    ):
        config = self.config
        env, hardware, cluster_config = self.env, self.hardware, self.cluster_config
        block_size = config.block_size
        operator_backend = config.operator_backend
        latency_fallback = config.latency_fallback
        latency_backend_type = config.latency_backend_type
        random_seed = config.random_seed
        af_dtype = af_dtype or self.model_spec.dtype
        self.af_runtime_plan: AFRuntimePlan | None = (
            compile_af_runtime_plan(
                cluster_config=cluster_config,
                worker_configs=worker_configs,
                model=self.model_spec,
                hardware=hardware,
                block_size=block_size,
                dtype=af_dtype,
                usable_memory_fraction=config.gpu_memory_utilization,
            )
            if cluster_config.af.enabled
            else None
        )
        self.af_replica_block_managers: dict[tuple[str, int], ReplicaKVBlockManager] = (
            {}
        )
        self.af_resources = (
            AFResourceRegistry(env, cluster_config, worker_configs, self.placement)
            if cluster_config.af.enabled
            else None
        )
        self.af_executor: AFExecutor | None = None
        self.af_iteration_summaries: list[dict[str, Any]] = []
        self.workers: list[LLMWorker | AFComputeWorker] = []
        self.request_workers: list[LLMWorker] = []
        for worker_id, worker_config in enumerate(worker_configs):
            physical_identity = (
                worker_config.physical_worker_ref.identity
                if worker_config.physical_worker_ref is not None
                else None
            )
            runtime_bindings = (
                self.af_runtime_plan.bindings_for_physical(physical_identity)
                if self.af_runtime_plan is not None and physical_identity is not None
                else ()
            )
            runtime_state = (
                self.af_runtime_plan.physical_state(physical_identity)
                if runtime_bindings and self.af_runtime_plan is not None
                else None
            )
            if self._is_request_owning_config(worker_config):
                has_af_scheduler_binding = any(
                    binding.is_scheduler_leader for binding in runtime_bindings
                )
                scheduler_replicas = (
                    self.af_runtime_plan.scheduler_replicas_for_physical(
                        physical_identity
                    )
                    if self.af_runtime_plan is not None
                    and physical_identity is not None
                    and has_af_scheduler_binding
                    else ()
                )
                replica = scheduler_replicas[0] if scheduler_replicas else None
                if replica is not None:
                    block_manager = next(
                        (
                            self.af_replica_block_managers[item.replica_key]
                            for item in scheduler_replicas
                            if item.replica_key in self.af_replica_block_managers
                        ),
                        None,
                    )
                    if block_manager is None:
                        block_manager = ReplicaKVBlockManager.from_plan(
                            self.af_runtime_plan,
                            replica,
                            aliases=scheduler_replicas,
                        )
                    for item in scheduler_replicas:
                        self.af_replica_block_managers[item.replica_key] = block_manager
                    cache_config_override = AFCacheConfig.from_plan(
                        self.af_runtime_plan, replica
                    )
                    worker_parallel_config = self.af_runtime_plan.domain_runtime(
                        replica.domain
                    ).parallel_config()
                else:
                    block_manager = None
                    cache_config_override = None
                    worker_parallel_config = self.parallel_config
                worker = LLMWorker(
                    env=env,
                    id=worker_id,
                    engine=self,
                    worker_config=worker_config,
                    kv_transfer_config=kv_transfer_config,
                    hardware=hardware,
                    model_spec=self.model_spec,
                    placement=self.placement,
                    parallel_config=worker_parallel_config,
                    moe_config=self.moe_config,
                    expert_placement=self.expert_placement,
                    kv_cache_capacity_tokens_per_dp_rank=cluster_config.effective_kv_cache_capacity_per_dp_rank(
                        self.parallel_config
                    ),
                    runtime_role=self._runtime_role(worker_config),
                    cache_config_override=cache_config_override,
                    block_manager=block_manager,
                    runtime_bindings=runtime_bindings,
                    runtime_state=runtime_state,
                )
                self.request_workers.append(worker)
            else:
                worker = AFComputeWorker(
                    env,
                    worker_id,
                    worker_config,
                    runtime_bindings=runtime_bindings,
                    runtime_state=runtime_state,
                )
            self.workers.append(worker)
        for worker in self.request_workers:
            worker.connector.bind_runtime(worker)
            worker.parallel_communicator.workers = self.workers

        if cluster_config.af.enabled:
            af_compute_provider, af_transfer_provider = self._resolve_af_providers(
                af_compute_provider,
                af_transfer_provider,
                worker_configs,
                af_dtype=af_dtype,
                operator_backend=operator_backend,
                latency_fallback=latency_fallback,
                latency_backend_type=latency_backend_type,
            )
            self.af_compute_provider = af_compute_provider
            self.af_transfer_provider = af_transfer_provider
            executor_type = ModelAFExecutor
            self.af_executor = executor_type(
                env=env,
                cluster_config=cluster_config,
                worker_configs=worker_configs,
                compute_provider=af_compute_provider,
                transfer_provider=af_transfer_provider,
                model_id=self.model_spec.model_id,
                dtype=af_dtype,
                total_layers=self.model_spec.num_layers,
                hidden_size=self.model_spec.hidden_size,
                resources=self.af_resources,
                moe_config=self.moe_config,
                random_seed=random_seed,
                runtime_plan=self.af_runtime_plan,
                model=self.model_spec,
            )

    def _resolve_af_providers(
        self,
        compute_provider: AFComputeLatencyProvider | None,
        transfer_provider: AFTransferLatencyProvider | None,
        worker_configs: list[WorkerConfig],
        *,
        af_dtype: str,
        operator_backend: str | None,
        latency_fallback: str,
        latency_backend_type: str,
    ) -> tuple[AFComputeLatencyProvider | None, AFTransferLatencyProvider | None]:
        """Build operator-table providers unless the caller injected both."""
        af_config = self.cluster_config.af
        if af_config.latency_provider != "operator_table" or (
            compute_provider is not None and transfer_provider is not None
        ):
            return compute_provider, transfer_provider
        from TokenSim.af.operator_provider import build_operator_table_af_providers

        built_compute, built_transfer = build_operator_table_af_providers(
            hardware=self.hardware,
            model=self.model_spec,
            cluster_config=self.cluster_config,
            worker_configs=worker_configs,
            placement=self.placement,
            dtype=af_dtype,
            operator_backend=af_config.operator_backend or operator_backend,
            fallback=af_config.latency_fallback or latency_fallback,
            latency_backend_type=latency_backend_type,
        )
        return compute_provider or built_compute, transfer_provider or built_transfer

    def _is_request_owning_config(self, worker_config: WorkerConfig) -> bool:
        return any(
            self._config_supports_phase(worker_config, phase)
            for phase in ("prefill", "decode")
        )

    def _config_supports_phase(
        self,
        worker_config: WorkerConfig,
        phase: str,
    ) -> bool:
        if self.cluster_config.af.enabled and phase in self.cluster_config.af.phases:
            physical = worker_config.physical_worker_ref
            return physical is not None and physical.identity in (
                self.cluster_config.af_request_owner_identities(phase)
            )
        capability = "prefill" if phase == "prefill" else "decode_full"
        return capability in worker_config.capabilities

    def _runtime_role(self, worker_config: WorkerConfig) -> str:
        supports_prefill = self._config_supports_phase(worker_config, "prefill")
        supports_decode = self._config_supports_phase(worker_config, "decode")
        if supports_prefill and supports_decode:
            return "hybrid"
        if supports_prefill:
            return "prefill"
        if supports_decode:
            return "decode"
        raise ConfigurationError(
            f"request-owning worker group {worker_config.name!r} is not bound to "
            "Prefill or Decode"
        )

    def _worker_supports_phase(self, worker: LLMWorker, phase: str) -> bool:
        if self.cluster_config.af.enabled and phase in self.cluster_config.af.phases:
            physical = worker.physical_worker_ref
            return physical is not None and physical.identity in (
                self.cluster_config.af_request_owner_identities(phase)
            )
        capability = "prefill" if phase == "prefill" else "decode_full"
        return capability in worker.capabilities

    def uses_af_for(self, requests: list[Request]) -> bool:
        return (
            self.af_executor is not None
            and bool(requests)
            and all(
                ("prefill" if req.is_prefill else "decode")
                in self.cluster_config.af.phases
                for req in requests
            )
        )

    def run_complete_compute(self, worker_id: int, latency: float):
        if self.af_resources is None:
            yield self.env.timeout(latency)
            return
        resource = self.af_resources.compute_slots[worker_id]
        with resource.request() as request:
            yield request
            yield self.env.timeout(latency)

    def record_af_iteration(self, iteration: AFIteration) -> None:
        """Keep aggregates here; complete trace events remain owned by the recorder."""
        if self.af_executor is None:
            return
        self.af_iteration_summaries.append(self.af_executor.trace.summarize(iteration))

    def af_runtime_summary(self) -> dict[str, Any] | None:
        if self.af_runtime_plan is None:
            return None
        return self.af_runtime_plan.summary(self.af_replica_block_managers)
