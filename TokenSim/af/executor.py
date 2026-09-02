from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import simpy

from TokenSim.af.builder import AFBlockBuilder, phase_for_requests
from TokenSim.af.latency import (
    AFComputeLatencyProvider,
    AFComputeQuery,
    AFServiceEstimate,
    AFTransferLatencyProvider,
    AFTransferQuery,
)
from TokenSim.af.models import AFBlock, AFIteration, AFPhase
from TokenSim.af.resources import AFResourceRegistry, AFWorkerToken
from TokenSim.af.trace import AFTraceEvent, AFTraceRecorder
from TokenSim.config.cluster_config import AFLinkConfig, ClusterConfig, WorkerConfig
from TokenSim.errors import ConfigurationError, SimulationStateError


class AFExecutor:
    def __init__(
        self,
        env: simpy.Environment,
        cluster_config: ClusterConfig,
        worker_configs: list[WorkerConfig],
        compute_provider: AFComputeLatencyProvider | None,
        transfer_provider: AFTransferLatencyProvider | None,
        model_id: str,
        dtype: str,
        total_layers: int,
        hidden_size: int,
        resources: AFResourceRegistry | None = None,
        trace_recorder: AFTraceRecorder | None = None,
    ) -> None:
        self._validate_runtime(
            cluster_config,
            compute_provider,
            transfer_provider,
            model_id,
            dtype,
            total_layers,
            hidden_size,
        )
        self.env = env
        self.cluster_config = cluster_config
        self.compute_provider = compute_provider
        self.transfer_provider = transfer_provider
        self.model_id = model_id
        self.dtype = dtype
        self.total_layers = total_layers
        self.hidden_size = hidden_size
        self.resources = resources or AFResourceRegistry(
            env, cluster_config, worker_configs
        )
        self.trace = trace_recorder or AFTraceRecorder(
            cluster_config.af.trace_enabled,
            self.resources.resource_capacities(),
        )
        self.builder = AFBlockBuilder()
        self._links = {link.name: link for link in cluster_config.af.links}
        self._next_iteration_id = 0
        self.active_iterations: dict[int, AFIteration] = {}
        self.active_stages: dict[tuple[int, int], dict[str, object]] = {}

    @staticmethod
    def _validate_runtime(
        cluster_config: ClusterConfig,
        compute_provider: AFComputeLatencyProvider | None,
        transfer_provider: AFTransferLatencyProvider | None,
        model_id: str,
        dtype: str,
        total_layers: int,
        hidden_size: int,
    ) -> None:
        if not cluster_config.af.enabled:
            raise ConfigurationError("AFExecutor requires af.enabled=true")
        missing = []
        if compute_provider is None:
            missing.append("compute latency provider")
        if transfer_provider is None:
            missing.append("transfer latency provider")
        if not isinstance(model_id, str) or not model_id:
            missing.append("model_id")
        if not isinstance(dtype, str) or not dtype:
            missing.append("dtype")
        if type(total_layers) is not int or total_layers < 1:
            missing.append("positive total_layers")
        if type(hidden_size) is not int or hidden_size < 1:
            missing.append("positive hidden_size")
        if missing:
            raise ConfigurationError(
                "AF runtime is missing required dependencies: " + ", ".join(missing)
            )

    def submit(
        self,
        source_a_worker_id: int,
        requests: Sequence[Any],
        phase: AFPhase | None = None,
    ) -> AFIteration:
        resolved_phase = phase or phase_for_requests(requests)
        if resolved_phase not in self.cluster_config.af.phases:
            raise ConfigurationError(
                f"AF phase {resolved_phase!r} is not selected by af.phases"
            )
        link = self.cluster_config.phase_to_af_link[resolved_phase]
        source_group = self.resources.group_for_worker(source_a_worker_id)
        if source_group != link.attention_group:
            raise ConfigurationError(
                f"AF phase {resolved_phase!r} requires attention group "
                f"{link.attention_group!r}, but worker {source_a_worker_id} belongs to "
                f"{source_group!r}"
            )
        iteration_id = self._next_iteration_id
        self._next_iteration_id += 1
        return self.builder.build(
            iteration_id=iteration_id,
            requests=requests,
            af_block_size=self.cluster_config.af.af_block_size,
            source_a_worker_id=source_a_worker_id,
            link=link,
            phase=resolved_phase,
        )

    def run_iteration(self, iteration: AFIteration):
        if iteration.iteration_id in self.active_iterations:
            raise SimulationStateError(
                f"AF iteration {iteration.iteration_id} is already active"
            )
        iteration.start_at = self.env.now
        self.active_iterations[iteration.iteration_id] = iteration
        failure_event = self.env.event()
        failures: list[BaseException] = []
        done_events = [self.env.event() for _ in iteration.blocks]
        processes = [
            self.env.process(
                self._run_block_guarded(block, failure_event, failures, done_event)
            )
            for block, done_event in zip(iteration.blocks, done_events)
        ]
        try:
            if processes:
                completion = self.env.all_of(done_events)
                yield self.env.any_of((completion, failure_event))
                if failures:
                    for process in processes:
                        if process.is_alive:
                            process.interrupt(
                                f"AF iteration {iteration.iteration_id} failed"
                            )
                    yield self.env.all_of(done_events)
                    exc = failures[0]
                    if isinstance(exc, SimulationStateError):
                        raise exc
                    raise SimulationStateError(
                        f"AF iteration {iteration.iteration_id} failed: {exc}"
                    ) from exc
        except simpy.Interrupt:
            for process in processes:
                if process.is_alive:
                    process.interrupt(
                        f"AF iteration {iteration.iteration_id} cancelled"
                    )
            if processes:
                yield self.env.all_of(done_events)
            raise
        finally:
            self.active_iterations.pop(iteration.iteration_id, None)
            for key in tuple(self.active_stages):
                if key[0] == iteration.iteration_id:
                    self.active_stages.pop(key, None)

        iteration.end_at = self.env.now
        iteration.barrier_wait = {
            block_id: iteration.end_at - completed_at
            for block_id, completed_at in iteration.block_completed_at.items()
        }
        return iteration

    def _run_block_guarded(
        self,
        block: AFBlock,
        failure_event: simpy.Event,
        failures: list[BaseException],
        done_event: simpy.Event,
    ):
        try:
            yield from self._run_block(block)
        except simpy.Interrupt:
            return
        except BaseException as exc:
            failures.append(exc)
            if not failure_event.triggered:
                failure_event.succeed()
            return
        finally:
            if not done_event.triggered:
                done_event.succeed()

    def _run_block(self, block: AFBlock):
        link = self._links[block.link_name]
        a_worker = self.resources.worker(block.source_a_worker_id)
        f_group_worker = self.resources.first_group_worker(link.ffn_group)
        for layer_id in range(self.total_layers):
            yield from self._run_a_compute(block, layer_id, a_worker.hardware)
            yield from self._run_transfer(
                block,
                layer_id,
                link,
                "a_to_f",
                a_worker.hardware,
                f_group_worker.hardware,
            )
            f_token = yield from self._run_f_compute(block, layer_id, link.ffn_group)
            yield from self._run_transfer(
                block,
                layer_id,
                link,
                "f_to_a",
                f_token.hardware,
                a_worker.hardware,
            )
        iteration = self.active_iterations[block.iteration_id]
        iteration.block_completed_at[block.block_id] = self.env.now

    def _run_a_compute(self, block: AFBlock, layer_id: int, hardware: str):
        resource = self.resources.compute_slots[block.source_a_worker_id]
        query = AFComputeQuery(
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.phase,
            stage="a",
            token_count=block.token_count,
            context_lengths=block.context_lengths,
            model_id=self.model_id,
            dtype=self.dtype,
            device_type=hardware,
            worker_id=block.source_a_worker_id,
        )
        ready_at, start_at, end_at, estimate = yield from self._run_stage(
            block,
            layer_id,
            "a",
            resource,
            lambda: self._estimate(self.compute_provider, query, block, layer_id, "a"),
        )
        self._record_compute(
            block,
            layer_id,
            "a",
            block.source_a_worker_id,
            ready_at,
            start_at,
            end_at,
            estimate,
        )
        self.active_stages.pop((block.iteration_id, block.block_id), None)

    def _run_f_compute(self, block: AFBlock, layer_id: int, group_name: str):
        pool = self.resources.ffn_pool(group_name)
        ready_at = self.env.now
        self._set_active_stage(block, layer_id, "f", ready_at)
        get_event = pool.get()
        token: AFWorkerToken | None = None
        try:
            token = yield get_event
            resource = self.resources.compute_slots[token.worker_id]
            query = AFComputeQuery(
                iteration_id=block.iteration_id,
                block_id=block.block_id,
                layer_id=layer_id,
                phase=block.phase,
                stage="f",
                token_count=block.token_count,
                context_lengths=block.context_lengths,
                model_id=self.model_id,
                dtype=self.dtype,
                device_type=token.hardware,
                worker_id=token.worker_id,
            )
            ready_at, start_at, end_at, estimate = yield from self._run_stage(
                block,
                layer_id,
                "f",
                resource,
                lambda: self._estimate(
                    self.compute_provider, query, block, layer_id, "f"
                ),
                ready_at=ready_at,
            )
        finally:
            if token is not None:
                pool.put(token)
            elif not get_event.triggered:
                get_event.cancel()
        self._record_compute(
            block,
            layer_id,
            "f",
            token.worker_id,
            ready_at,
            start_at,
            end_at,
            estimate,
        )
        self.active_stages.pop((block.iteration_id, block.block_id), None)
        return token

    def _run_transfer(
        self,
        block: AFBlock,
        layer_id: int,
        link: AFLinkConfig,
        direction: str,
        source_device: str,
        target_device: str,
    ):
        key, resource = self.resources.communication_resource(link, direction)
        query = AFTransferQuery(
            iteration_id=block.iteration_id,
            block_id=block.block_id,
            layer_id=layer_id,
            phase=block.phase,
            direction=direction,
            token_count=block.token_count,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            source_device=source_device,
            target_device=target_device,
            network=link.network,
        )
        ready_at, start_at, end_at, estimate = yield from self._run_stage(
            block,
            layer_id,
            direction,
            resource,
            lambda: self._estimate(
                self.transfer_provider, query, block, layer_id, direction
            ),
        )
        self.trace.record(
            AFTraceEvent(
                iteration_id=block.iteration_id,
                block_id=block.block_id,
                layer_id=layer_id,
                phase=block.phase,
                event_type="transfer",
                direction=direction,
                resource_id=self.resources.format_link_resource(key),
                ready_at=ready_at,
                start_at=start_at,
                end_at=end_at,
                queue_duration=start_at - ready_at,
                service_duration=end_at - start_at,
                request_count=block.request_count,
                token_count=block.token_count,
                source_id=estimate.source_id,
                dataset_version=estimate.dataset_version,
                match_type=estimate.match_type,
                payload_bytes=estimate.payload_bytes,
                details=estimate.details,
            )
        )
        self.active_stages.pop((block.iteration_id, block.block_id), None)

    def _run_stage(
        self,
        block: AFBlock,
        layer_id: int,
        stage: str,
        resource: simpy.Resource,
        estimate_factory,
        ready_at: float | None = None,
    ):
        ready_at = self.env.now if ready_at is None else ready_at
        self._set_active_stage(block, layer_id, stage, ready_at)
        with resource.request() as request:
            yield request
            start_at = self.env.now
            self._set_active_stage(block, layer_id, stage, ready_at, start_at)
            estimate = estimate_factory()
            yield self.env.timeout(estimate.service_duration)
            end_at = self.env.now
        return ready_at, start_at, end_at, estimate

    def _record_compute(
        self,
        block: AFBlock,
        layer_id: int,
        stage: str,
        worker_id: int,
        ready_at: float,
        start_at: float,
        end_at: float,
        estimate: AFServiceEstimate,
    ) -> None:
        self.trace.record(
            AFTraceEvent(
                iteration_id=block.iteration_id,
                block_id=block.block_id,
                layer_id=layer_id,
                phase=block.phase,
                event_type="compute",
                stage=stage,
                resource_id=f"worker:{worker_id}",
                worker_id=worker_id,
                ready_at=ready_at,
                start_at=start_at,
                end_at=end_at,
                queue_duration=start_at - ready_at,
                service_duration=end_at - start_at,
                request_count=block.request_count,
                token_count=block.token_count,
                source_id=estimate.source_id,
                dataset_version=estimate.dataset_version,
                match_type=estimate.match_type,
                details=estimate.details,
            )
        )

    @staticmethod
    def _estimate(provider, query, block, layer_id, stage) -> AFServiceEstimate:
        try:
            estimate = provider.estimate(query)
        except Exception as exc:
            raise SimulationStateError(
                f"AF estimate unsupported for iteration={block.iteration_id}, "
                f"block={block.block_id}, layer={layer_id}, stage={stage}: {exc}"
            ) from exc
        if not isinstance(estimate, AFServiceEstimate):
            raise SimulationStateError(
                f"AF provider returned {type(estimate).__name__} for "
                f"iteration={block.iteration_id}, block={block.block_id}, "
                f"layer={layer_id}, stage={stage}; expected AFServiceEstimate"
            )
        duration = estimate.service_duration
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            valid = False
        else:
            valid = math.isfinite(duration) and duration >= 0
        if not valid:
            raise SimulationStateError(
                f"AF provider returned invalid service_duration={duration!r} for "
                f"iteration={block.iteration_id}, block={block.block_id}, "
                f"layer={layer_id}, stage={stage}"
            )
        if not isinstance(estimate.source_id, str) or not estimate.source_id:
            raise SimulationStateError(
                f"AF provider returned an empty source_id for iteration="
                f"{block.iteration_id}, block={block.block_id}, layer={layer_id}, "
                f"stage={stage}"
            )
        if (
            not isinstance(estimate.dataset_version, str)
            or not estimate.dataset_version
        ):
            raise SimulationStateError(
                f"AF provider returned an empty dataset_version for iteration="
                f"{block.iteration_id}, block={block.block_id}, layer={layer_id}, "
                f"stage={stage}"
            )
        if estimate.match_type not in {"exact", "interpolated"}:
            raise SimulationStateError(
                f"AF provider returned invalid match_type={estimate.match_type!r} "
                f"for iteration={block.iteration_id}, block={block.block_id}, "
                f"layer={layer_id}, stage={stage}"
            )
        return estimate

    def _set_active_stage(
        self,
        block: AFBlock,
        layer_id: int,
        stage: str,
        ready_at: float,
        start_at: float | None = None,
    ) -> None:
        self.active_stages[(block.iteration_id, block.block_id)] = {
            "iteration_id": block.iteration_id,
            "block_id": block.block_id,
            "layer_id": layer_id,
            "stage": stage,
            "ready_at": ready_at,
            "start_at": start_at,
        }

    def unfinished_snapshot(self) -> list[dict[str, object]]:
        snapshots: list[dict[str, object]] = []
        for iteration in self.active_iterations.values():
            snapshots.append(
                {
                    "iteration_id": iteration.iteration_id,
                    "phase": iteration.phase,
                    "block_ids": [block.block_id for block in iteration.blocks],
                    "completed_block_ids": sorted(iteration.block_completed_at),
                    "active_stages": [
                        dict(stage)
                        for (iteration_id, _), stage in self.active_stages.items()
                        if iteration_id == iteration.iteration_id
                    ],
                }
            )
        return snapshots
