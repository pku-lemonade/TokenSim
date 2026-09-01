from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import simpy

from TokenSim.llm.llm_request import LLMTime, RequestTime, g_time
from TokenSim.llm.llm_scheduler import LLMDynamicScheduler
from TokenSim.block.kv_cache_manager import PrefixReusePlan
from TokenSim.block.block_table import BlockTable
from TokenSim.block.block_manager import BlockManager
from TokenSim.block.block import Device, PhysicalTokenBlock
from TokenSim.config.config import KVTransferConfig
from TokenSim.kv_transfer import (
    ConnectorStats,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
    NoopConnector,
    P2PConnector,
)
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.latency import RooflineLatencyBackend, build_latency_backend
from TokenSim.llm.llm_engine import (
    LLMEngine,
    LLMWorker,
    RequestCompletionDebugPrinter,
    Task,
)
from TokenSim.llm.llm_request import Request, RequestStatus
from TokenSim.llm.llm_scheduler import LLMPagedAttnScheduler
from TokenSim.config.config import ParallelConfig, WorkerGroupConfig
from TokenSim.placement import (
    BalancedLoadWorkerPool,
    LeastGpuMemoryWorkerPool,
    RoundRobinWorkerPool,
)
from TokenSim.timing import TimingRecorder
from benchmark import check_results, get_latency_backend_type
from util.results import print_slo_stats


class TimingRecorderTest(unittest.TestCase):
    def test_timing_recorder_reset_clears_compatible_storage(self):
        timing = LLMTime()
        recorder = TimingRecorder(timing)
        recorder.record_request(
            RequestTime.from_step_lists(
                id=1,
                time=[0.1, 0.2],
                service_time=[0.1, 0.2],
                batch=[1, 1],
            )
        )

        self.assertEqual(len(timing.time), 1)
        recorder.reset()
        self.assertEqual(timing.time, [])

    def test_slo_stats_uses_passed_timing(self):
        g_time.time.clear()
        timing = LLMTime()
        timing.time.append(
            RequestTime.from_step_lists(
                id=1,
                time=[1.0, 0.1],
                service_time=[1.0, 0.1],
                batch=[1, 1],
            )
        )

        output = io.StringIO()
        with redirect_stdout(output):
            print_slo_stats(2.0, timing, prefill_slo=2.0, decode_slo=0.2)

        self.assertIn("Prefill SLO Good Throughput: 1 r, 0.5 r/s", output.getvalue())
        self.assertEqual(g_time.time, [])


class _PlacementWorker:
    def __init__(
        self, id: int, workload: float, cpu_workload: float = 0, scheduler=None
    ):
        self.id = id
        self._workload = workload
        self._cpu_workload = cpu_workload
        self.scheduler = scheduler or _PlacementScheduler()

    def workload(self) -> float:
        return self._workload

    def cpu_workload(self) -> float:
        return self._cpu_workload


class _PlacementScheduler:
    def __init__(self, running=0, waiting=0):
        self.running = [object()] * running
        self.waiting = [object()] * waiting


class PlacementPolicyTest(unittest.TestCase):
    def test_round_robin_pool_schedules_round_robin(self):
        workers = [_PlacementWorker(3, 0), _PlacementWorker(4, 0)]
        pool = RoundRobinWorkerPool(workers)

        self.assertEqual([pool.schedule().id for _ in range(4)], [3, 4, 3, 4])
        self.assertIs(pool.get(4), workers[1])

    def test_placement_selection_apis_count_decisions(self):
        workers = [_PlacementWorker(3, 0), _PlacementWorker(4, 0)]
        pool = RoundRobinWorkerPool(workers)

        self.assertEqual(pool.select_prefill_worker().id, 3)
        self.assertEqual(pool.select_decode_worker().id, 4)
        self.assertEqual(pool.select_transfer_target().id, 3)
        self.assertEqual(pool.placement_decision_count, 3)

    def test_least_gpu_memory_pool_schedules_lowest_workload(self):
        workers = [
            _PlacementWorker(3, 70),
            _PlacementWorker(4, 10),
            _PlacementWorker(5, 50),
        ]
        pool = LeastGpuMemoryWorkerPool(workers)

        self.assertIs(pool.schedule(), workers[1])

    def test_balanced_load_pool_considers_active_requests_before_memory(self):
        workers = [
            _PlacementWorker(3, 1, scheduler=_PlacementScheduler(running=3)),
            _PlacementWorker(4, 90, scheduler=_PlacementScheduler(running=1)),
            _PlacementWorker(5, 10, scheduler=_PlacementScheduler(running=2)),
        ]
        pool = BalancedLoadWorkerPool(workers)

        self.assertIs(pool.schedule(), workers[1])

    def test_worker_group_accepts_current_worker_roles(self):
        networks = {"net1": "ethernet100Gb"}
        for role in ("prefill", "decode", "hybrid"):
            with self.subTest(role=role):
                config = WorkerGroupConfig(
                    role=role,
                    hardware="A100-40G",
                    num_workers=1,
                    network="net1",
                )
                workers = config.workers(networks)

                self.assertEqual(workers[0].role, role)


class _RooflineStub:
    def __init__(self):
        self.calls = []

    def Compute_Timebreakdown_Iteration(
        self,
        prefill_len,
        generation_idx,
        batch_size,
        model,
        hardware,
        Pipeline_Stage,
    ):
        self.calls.append(
            (prefill_len, generation_idx, batch_size, model, hardware, Pipeline_Stage)
        )
        return 0.01, 0.001


class _LatencyRequest:
    def __init__(
        self,
        prefill_len: int,
        generation_idx: int,
        is_prefill: bool,
        prefill_compute_len: int | None = None,
        needs_recompute: bool = False,
        recompute_tokens: int = 0,
    ):
        self.prefill_len = prefill_len
        self.generation_idx = generation_idx
        self.is_prefill = is_prefill
        self.prefill_compute_len = (
            prefill_len if prefill_compute_len is None else prefill_compute_len
        )
        self.needs_recompute = needs_recompute
        self.recompute_tokens = recompute_tokens


class LatencyBackendTest(unittest.TestCase):
    def test_latency_backend_cli_uses_roofline_or_llmcompass_template_path(self):
        self.assertEqual(get_latency_backend_type("roofline"), "roofline")
        self.assertEqual(
            get_latency_backend_type("./LLMCompass/examples/a100.json"),
            "llm_compass",
        )

    def test_latency_backend_builder_creates_roofline_backend(self):
        backend = build_latency_backend(
            "roofline",
            roofline=_RooflineStub(),
            model="model",
            hardware="hardware",
            parallel_config=ParallelConfig(),
        )

        self.assertIsInstance(backend, RooflineLatencyBackend)

    def test_latency_backend_builder_requires_llmcompass_vars(self):
        with self.assertRaises(ConfigurationError):
            build_latency_backend(
                "llm_compass",
                roofline=_RooflineStub(),
                model="model",
                hardware="hardware",
                parallel_config=ParallelConfig(),
            )

    def test_roofline_backend_prefill_projection_uses_packed_context_batch(self):
        roofline = _RooflineStub()
        backend = RooflineLatencyBackend(
            roofline, "model", "hardware", ParallelConfig()
        )
        requests = [
            _LatencyRequest(prefill_len=64, generation_idx=0, is_prefill=True),
            _LatencyRequest(prefill_len=64, generation_idx=0, is_prefill=True),
        ]

        backend.estimate_step_latency(requests)

        self.assertEqual(roofline.calls[0][0], 256)
        self.assertEqual(roofline.calls[0][2], 1)

    def test_roofline_backend_uses_nonzero_attention_prefill_for_zero_effective_prompt(
        self,
    ):
        roofline = _RooflineStub()
        backend = RooflineLatencyBackend(
            roofline, "model", "hardware", ParallelConfig()
        )
        request = _LatencyRequest(
            prefill_len=128,
            prefill_compute_len=0,
            generation_idx=0,
            is_prefill=True,
        )

        backend.estimate_step_latency([request])

        self.assertEqual(roofline.calls[1][0], 1)

    def test_roofline_backend_decode_uses_original_prefill_len_without_hits(self):
        roofline = _RooflineStub()
        backend = RooflineLatencyBackend(
            roofline, "model", "hardware", ParallelConfig()
        )
        request = _LatencyRequest(prefill_len=128, generation_idx=1, is_prefill=False)

        backend.estimate_step_latency([request])

        # The projection and per-request attention lookups share the same
        # (prompt_len, step, batch) key here, so memoization dedupes them into
        # a single roofline call — still with the original prefill length.
        self.assertEqual(len(roofline.calls), 1)
        self.assertEqual(roofline.calls[0][0], 128)

    def test_roofline_backend_models_recompute_as_context_build(self):
        roofline = _RooflineStub()
        backend = RooflineLatencyBackend(
            roofline, "model", "hardware", ParallelConfig()
        )
        request = _LatencyRequest(
            prefill_len=128,
            generation_idx=17,
            is_prefill=False,
            needs_recompute=True,
            recompute_tokens=96,
        )

        backend.estimate_step_latency([request])

        self.assertEqual(roofline.calls[0][0], 128)
        self.assertEqual(roofline.calls[0][1], 0)


class SchedulerOutputTest(unittest.TestCase):
    def test_dynamic_scheduler_output_includes_connector_metadata(self):
        scheduler = LLMDynamicScheduler(max_parallem_sum=10)
        request = _LatencyRequest(prefill_len=5, generation_idx=0, is_prefill=True)
        request.arrival_time = 0
        scheduler.add_requests([request])

        output = scheduler.schedule_output()

        self.assertEqual(output.running, [request])
        self.assertEqual(output.preempted, [])
        self.assertEqual(output.scheduled, [request])
        self.assertIsInstance(output.connector_metadata, KVConnectorMetadata)


def _request(request_id: int, prefill_len: int = 16, decode_len: int = 2) -> Request:
    req = Request(
        id=request_id,
        prefill_len=prefill_len,
        decode_len=decode_len,
        block_size=16,
    )
    req.arrive(type("Env", (), {"now": float(request_id)})())
    return req


def _cache_config(num_gpu_blocks: int, num_cpu_blocks: int = 16):
    return type(
        "CacheConfigStub",
        (),
        {
            "block_size": 16,
            "num_gpu_blocks": num_gpu_blocks,
            "num_cpu_blocks": num_cpu_blocks,
            "model": "test",
        },
    )()


class PagedAttentionSchedulerTest(unittest.TestCase):
    def test_admission_respects_max_parallel_sum(self):
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=_cache_config(num_gpu_blocks=16),
            max_parallem_sum=1,
            max_occupy_ratio=1,
        )
        req0 = _request(0)
        req1 = _request(1)
        scheduler.add_requests([req0, req1])

        output = scheduler.schedule_output()

        self.assertEqual(output.scheduled, [req0])
        self.assertEqual(scheduler.waiting, [req1])

    def test_admission_respects_max_occupancy(self):
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=_cache_config(num_gpu_blocks=16),
            max_parallem_sum=10,
            max_occupy_ratio=0,
        )
        scheduler.add_requests([_request(0)])

        output = scheduler.schedule_output()

        self.assertEqual(output.running, [])
        self.assertEqual(len(scheduler.waiting), 1)

    def test_full_cache_without_new_logical_block_does_not_preempt(self):
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=_cache_config(num_gpu_blocks=2, num_cpu_blocks=8),
            max_parallem_sum=10,
            max_occupy_ratio=1,
        )
        req0 = _request(0, prefill_len=16, decode_len=2)
        req1 = _request(1, prefill_len=16, decode_len=2)
        scheduler.add_requests([req0, req1])
        scheduler.schedule_output()

        output = scheduler.schedule_output()

        self.assertEqual(output.preempted, [])
        self.assertEqual(output.running, [req0, req1])

    def test_preemption_requeues_and_recomputes_request(self):
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=_cache_config(num_gpu_blocks=2, num_cpu_blocks=8),
            max_parallem_sum=10,
            max_occupy_ratio=1,
        )
        req0 = _request(0, prefill_len=16, decode_len=3)
        req1 = _request(1, prefill_len=16, decode_len=3)
        scheduler.add_requests([req0, req1])
        scheduler.schedule_output()
        for req in (req0, req1):
            req.generation_idx = 1
            req._append_tokens(1)

        output = scheduler.schedule_output()

        self.assertEqual(output.preempted, [req1])
        self.assertEqual(req1.status, RequestStatus.WAITING)
        self.assertTrue(req1.needs_recompute)
        self.assertEqual(scheduler.waiting, [req1])
        self.assertIn(req1.id, output.connector_metadata.preempted_request_ids)

        req0.generation_idx = req0.decode_len
        scheduler.update([req0])
        resumed = scheduler.schedule_output()

        self.assertEqual(resumed.running, [req1])
        self.assertTrue(req1.needs_recompute)
        scheduler.finish_recompute([req1], latency=0.25)
        self.assertFalse(req1.needs_recompute)
        self.assertEqual(req1.generation_idx, 1)
        self.assertEqual(req1.recomputation_count, 1)
        self.assertEqual(req1.recomputed_tokens_total, 17)
        self.assertEqual(req1.recompute_service_time, 0.25)

    def test_request_with_available_slot_reenters_running(self):
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=_cache_config(num_gpu_blocks=3, num_cpu_blocks=8),
            max_parallem_sum=10,
            max_occupy_ratio=1,
        )
        req = _request(0, prefill_len=16, decode_len=2)
        req.status = RequestStatus.RUNNING
        scheduler.running = [req]
        scheduler.block_manager.allocate(req)

        output = scheduler.schedule_output()

        self.assertIn(req, output.running)
        self.assertEqual(req.status, RequestStatus.RUNNING)


class SimulationLivenessGuardTest(unittest.TestCase):
    def test_capacity_validation_rejects_oversized_decode_context(self):
        scheduler = LLMPagedAttnScheduler(
            id=0,
            cache_config=_cache_config(num_gpu_blocks=4, num_cpu_blocks=8),
            max_parallem_sum=10,
            max_occupy_ratio=1,
        )
        worker = SimpleNamespace(scheduler=scheduler)
        engine = object.__new__(LLMEngine)
        engine.prefill_workers = SimpleNamespace(workers=[worker])
        engine.decode_workers = SimpleNamespace(workers=[worker])

        with self.assertRaisesRegex(ConfigurationError, "required_blocks=8"):
            engine.validate_request_capacity(
                [_request(7, prefill_len=16, decode_len=112)]
            )

    def test_incomplete_simulation_writes_failure_json_and_raises(self):
        request = _request(3, prefill_len=16, decode_len=2)
        engine = SimpleNamespace(workers=[])
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                cluster="cluster.json",
                qps=2.0,
                results_path=directory,
            )

            with self.assertRaisesRegex(SimulationStateError, "unfinished requests"):
                check_results(
                    args=args,
                    requests=[request],
                    engine=engine,
                    model_config=None,
                    cluster=None,
                    duration=5.0,
                    request_count=1,
                    prefill_lens=[16],
                    decode_lens=[2],
                    simulator_wall_time=0.5,
                )

            failure_file = Path(directory) / "failure_2.0.json"
            snapshot = json.loads(failure_file.read_text())
            self.assertEqual(
                snapshot["failure"],
                "simpy_event_queue_exhausted_with_unfinished_requests",
            )
            self.assertEqual(snapshot["unfinished_requests"][0]["id"], 3)
            self.assertFalse((Path(directory) / "result_2.0.json").exists())


class PrefixReusePlanTest(unittest.TestCase):
    def test_prefix_reuse_plan_is_immutable(self):
        plan = PrefixReusePlan(
            request_id=1,
            hit_blocks=(),
            input_keys=(),
            full_input_blocks=0,
            total_logical_blocks=1,
            has_hash_ids=False,
            hit_tokens=0,
            effective_prefill_tokens=16,
        )

        with self.assertRaises(Exception):
            plan.hit_tokens = 1


class BlockBoundaryTest(unittest.TestCase):
    def test_block_table_get_blocks_returns_copy(self):
        table = BlockTable()
        block = PhysicalTokenBlock(Device.GPU, 0, 16)
        table.add_block(1, block)

        external = table.get_blocks(1)
        external.clear()

        self.assertEqual(table.get_num_blocks(1), 1)

    def test_free_splits_cache_commit_and_release_steps(self):
        manager = BlockManager(
            block_size=16,
            num_gpu_blocks=4,
            num_cpu_blocks=4,
            model="llama",
            watermark=0,
        )
        req = _request(0, prefill_len=16, decode_len=1)
        manager.allocate(req)
        manager.commit_finished_cache(req)

        self.assertTrue(req.input_cache_committed)
        self.assertEqual(manager.block_table.get_num_blocks(req.id), 1)

        manager.release_request_blocks(req)

        self.assertEqual(manager.block_table.get_num_blocks(req.id), 0)


class _CacheConfigStub:
    block_size = 16
    size_per_token = 8


class ConnectorTest(unittest.TestCase):
    def test_noop_connector_builds_empty_metadata(self):
        connector = NoopConnector(KVTransferConfig.default())

        metadata = connector.build_connector_meta(object())

        self.assertEqual(metadata.connector_name, "NoopConnector")
        self.assertFalse(metadata.has_work)

    def test_p2p_connector_builds_transfer_plan_and_counts_bytes(self):
        req = _request(1)
        req._physical_token_blocks = [
            PhysicalTokenBlock(Device.GPU, 0, 16),
            PhysicalTokenBlock(Device.GPU, 1, 16),
        ]
        connector = P2PConnector(
            KVTransferConfig(kv_connector="P2PConnector"), _CacheConfigStub()
        )

        plan = connector.build_transfer_plan(
            requests=[req],
            source_worker_id=1,
            target_worker_id=2,
            kind="load",
            latency=0.25,
        )
        metadata = KVConnectorMetadata(connector_name="P2PConnector", loads=[plan])
        connector.bind_connector_metadata(metadata)

        self.assertEqual(plan.blocks, 2)
        self.assertEqual(plan.bytes, 2 * 16 * 8)
        self.assertEqual(connector.start_load_kv(), 0.25)
        self.assertEqual(connector.stats.transfer_count, 1)
        self.assertEqual(connector.stats.transfer_blocks, 2)
        self.assertEqual(connector.stats.transfer_bytes, 2 * 16 * 8)
        self.assertEqual(connector.stats.load_wait_time, 0.25)

    def test_connector_worker_metadata_aggregates(self):
        left = KVConnectorWorkerMetadata(finished_sending={1}, events=["a"])
        right = KVConnectorWorkerMetadata(finished_recving={2}, events=["b"])

        merged = left.aggregate(right)

        self.assertEqual(merged.finished_sending, {1})
        self.assertEqual(merged.finished_recving, {2})
        self.assertEqual(merged.events, ["a", "b"])

    def test_connector_stats_aggregate(self):
        stats = ConnectorStats(transfer_count=1, transfer_blocks=2)
        merged = stats.aggregate(ConnectorStats(transfer_count=3, transfer_bytes=4))

        self.assertEqual(merged.transfer_count, 4)
        self.assertEqual(merged.transfer_blocks, 2)
        self.assertEqual(merged.transfer_bytes, 4)


class RequestCompletionDebugPrinterTest(unittest.TestCase):
    def _worker(self, env, enabled: bool):
        printer = RequestCompletionDebugPrinter(enabled)
        printer.start(10.0)
        engine = type(
            "DebugEngineStub",
            (),
            {
                "record_request_completion": lambda self, event, request: printer.record(
                    event, request, env.now
                )
            },
        )()
        return type("DebugWorkerStub", (), {"env": env, "engine": engine})()

    def test_disabled_debug_print_is_silent(self):
        env = simpy.Environment()
        request = Request(id=7, prefill_len=32, decode_len=2, block_size=16)
        request.arrive(env)
        worker = self._worker(env, enabled=False)

        output = io.StringIO()
        with patch("TokenSim.llm.llm_engine.time.perf_counter", return_value=11.0):
            with redirect_stdout(output):
                LLMWorker.step(worker, [request], latency=0.1)

        self.assertEqual(output.getvalue(), "")

    def test_prefill_and_decode_records_use_wall_and_simulated_time(self):
        env = simpy.Environment()
        request = Request(id=42, prefill_len=1024, decode_len=2, block_size=16)
        request.arrive(env)
        worker = self._worker(env, enabled=True)
        output = io.StringIO()

        env.run(until=3.75)
        with patch(
            "TokenSim.llm.llm_engine.time.perf_counter",
            side_effect=[10.123456, 10.25],
        ):
            with redirect_stdout(output):
                LLMWorker.step(worker, [request], latency=0.1)
                env.run(until=4.0)
                LLMWorker.step(worker, [request], latency=0.1)

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "[request=42] debug event=prefill wall_elapsed_s=0.123456 request_id=42 "
                "input_len=1024 output_len=1 timestamp=3.750000",
                "[request=42] debug event=decode wall_elapsed_s=0.250000 request_id=42 "
                "input_len=1024 output_len=2 timestamp=4.000000",
            ],
        )

    def test_single_token_request_emits_both_events(self):
        env = simpy.Environment()
        request = Request(id=9, prefill_len=64, decode_len=1, block_size=16)
        request.arrive(env)
        worker = self._worker(env, enabled=True)
        output = io.StringIO()

        env.run(until=2.0)
        with patch(
            "TokenSim.llm.llm_engine.time.perf_counter",
            side_effect=[11.0, 11.0],
        ):
            with redirect_stdout(output):
                LLMWorker.step(worker, [request], latency=0.1)

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("event=prefill", lines[0])
        self.assertIn("event=decode", lines[1])
        self.assertTrue(
            all("output_len=1 timestamp=2.000000" in line for line in lines)
        )


class _EngineDispatchWorker:
    def __init__(self, env, worker_id: int):
        self.id = worker_id
        self.msg_queue = simpy.PriorityStore(env)
        self.connector = P2PConnector(
            KVTransferConfig(kv_connector="P2PConnector"),
            _CacheConfigStub(),
        )
        self.kv_transfer_config = KVTransferConfig(kv_connector="P2PConnector")
        self.cache_config = _CacheConfigStub()
        self.scheduler = type(
            "SchedulerStub",
            (),
            {
                "block_manager": type(
                    "BlockManagerStub",
                    (),
                    {"get_request_block_counts": lambda self, requests: {7: 2}},
                )()
            },
        )()

    def estimate_transfer_latency(self, remote_id: int, num_blocks: int) -> float:
        return 0.5


class EngineDispatchTest(unittest.TestCase):
    def test_prefill_decode_dispatch_builds_connector_metadata(self):
        env = simpy.Environment()
        engine = LLMEngine.__new__(LLMEngine)
        engine.id = -1
        sender = _EngineDispatchWorker(env, worker_id=1)
        receiver = _EngineDispatchWorker(env, worker_id=2)
        engine.decode_workers = RoundRobinWorkerPool([receiver])

        req = _request(7)
        req.status = RequestStatus.RUNNING
        sender.scheduler.block_manager.release_request_blocks = lambda request: None
        engine.dispatch_prefill_to_decode(sender, [req])

        message = receiver.msg_queue.items[0]
        self.assertEqual(message.task, Task.ADD)
        self.assertEqual(message.sender_id, -1)
        self.assertEqual(message.metadata.connector_name, "P2PConnector")
        self.assertEqual(message.metadata.loads[0].target_worker_id, 2)
        self.assertEqual(message.metadata.loads[0].blocks, 2)
        self.assertEqual(message.metadata.saves, [])
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_KV)

    def test_same_worker_prefill_decode_dispatch_has_no_load(self):
        env = simpy.Environment()
        engine = LLMEngine.__new__(LLMEngine)
        engine.id = -1
        worker = _EngineDispatchWorker(env, worker_id=1)
        engine.decode_workers = RoundRobinWorkerPool([worker])

        req = _request(7)
        req.status = RequestStatus.RUNNING
        worker.scheduler.block_manager.release_request_blocks = lambda request: None
        engine.dispatch_prefill_to_decode(worker, [req])

        message = worker.msg_queue.items[0]
        self.assertEqual(message.task, Task.ADD)
        self.assertEqual(message.metadata.loads, [])
        self.assertEqual(message.metadata.saves, [])


if __name__ == "__main__":
    unittest.main()
