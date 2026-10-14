"""Missing-shape reporting across compute and communication tables.

Covers the aggregated :class:`MissingShapeReport`, the fallback policy of the
collective model, the shared per-worker report exported by the engine, and the
AgentX-shaped acceptance checks against the shipped B300 vLLM package.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import simpy

from TokenSim.comm.collectives import CollectiveModel, CollectiveQuery, EPAllToAllQuery
from TokenSim.config.config import ClusterConfig, ParallelConfig, WorkerGroupConfig
from TokenSim.config.psla_config import MetricData, PSLAConfig
from TokenSim.errors import SimulationStateError
from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.links import LinkCatalog
from TokenSim.hardware.topology import TopologySpec
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, g_time, reset_g_time
from TokenSim.operator_data.coverage import MissingShapeReport
from TokenSim.config.model_config import ModelSpec
from TokenSim.latency import OperatorTableLatencyBackend
from TokenSim.operator_data.lookup import MissingOperatorDataError, OperatorLookup
from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta, SourceRecord
from TokenSim.operator_data.schema import ATTENTION_SEQ_EXTRAPOLATION_RATIO, TABLE_SPECS
from tests.hardware_fixtures import test_device, test_hardware, test_links, test_model, test_topology
from util.request import LLMSource
from util.results import analytical_share_by_table, export_result, get_latency_stats

_B300_VLLM = Path("data/operator_data/b300_sxm/vllm")


def _gemm_only_package(device_id: str = "TestGPU") -> OperatorDataPackage:
    source = SourceRecord("measured:test", "A", "measured", device=device_id, backend="test")
    meta = PackageMeta(dataset_version="test-v1", device_id=device_id, backend="test", sources={"measured:test": source})
    rows = [
        {"dtype": "fp16", "m": m, "n": n, "k": k, "latency_us": 10.0 + m, "source_id": "measured:test"}
        for m in (1, 8, 16, 64)
        for n in (16, 32, 48, 64, 1024)
        for k in (16, 32)
    ]
    return OperatorDataPackage.from_rows(meta, {"gemm": rows})


class MissingShapeReportTest(unittest.TestCase):
    def test_groups_by_discrete_key_and_tracks_axis_ranges(self):
        report = MissingShapeReport(max_examples=2)
        for m in (40_000, 90_000, 65_536):
            report.record("gemm", {"dtype": "bf16", "m": m, "n": 7168, "k": 7168}, f"m={m} beyond", "out_of_range")
        report.record("gemm", {"dtype": "fp8", "m": 40_000, "n": 7168, "k": 7168}, "m beyond", "out_of_range")
        report.record(
            "elementwise",
            {"op_name": "rmsnorm", "dtype": "bf16", "num_tokens": 16, "hidden_size": 7168},
            "table has no rows in this package",
            "table_absent",
        )
        report.record_error(
            MissingOperatorDataError(
                "ep_all2all",
                {"dtype": "bf16", "phase": "dispatch", "mode": "deepep_high_throughput", "ep_size": 64, "nodes": 8,
                 "hidden_size": 7168, "top_k": 8, "num_experts": 256, "num_tokens": 512},
                "no rows share the discrete key",
                kind="discrete_key",
            )
        )

        self.assertEqual((report.group_count, report.shape_count, report.query_count), (4, 6, 6))
        first = report.records()[0]
        self.assertEqual(first["table"], "gemm")
        self.assertEqual(first["key"], {"dtype": "bf16"})
        self.assertEqual(first["axes"], {"k": [7168, 7168], "m": [40_000, 90_000], "n": [7168, 7168]})
        self.assertEqual((first["query_count"], first["shape_count"], first["kinds"]), (3, 3, {"out_of_range": 3}))
        self.assertEqual(len(first["examples"]), 2)
        ep = next(g for g in report.records() if g["table"] == "ep_all2all")
        self.assertEqual(ep["key"], {"dtype": "bf16", "phase": "dispatch", "mode": "deepep_high_throughput", "ep_size": 64, "nodes": 8})
        self.assertEqual(report.per_table()["gemm"], {"groups": 2, "shapes": 4, "queries": 4})

    def test_merge_unions_groups_and_widens_axis_ranges(self):
        left = MissingShapeReport()
        right = MissingShapeReport()
        left.record("gemm", {"dtype": "bf16", "m": 100, "n": 8, "k": 8}, "r", "out_of_range")
        right.record("gemm", {"dtype": "bf16", "m": 900, "n": 8, "k": 8}, "r", "out_of_range")
        right.record("moe", {"dtype": "bf16", "distribution": "uniform", "num_tokens": 5, "hidden_size": 8,
                             "inter_size": 8, "top_k": 2, "num_experts": 4, "tp_size": 1, "ep_size": 4}, "r", "table_absent")

        merged = left.merge(right)

        self.assertEqual((merged.group_count, merged.shape_count, merged.query_count), (2, 3, 3))
        gemm = next(g for g in merged.records() if g["table"] == "gemm")
        self.assertEqual(gemm["axes"]["m"], [100, 900])
        self.assertEqual(merged.to_dict()["per_table"]["moe"]["queries"], 1)

    def test_group_cap_counts_dropped_queries(self):
        report = MissingShapeReport(max_groups=1)
        report.record("gemm", {"dtype": "bf16", "m": 1, "n": 8, "k": 8}, "r")
        report.record("gemm", {"dtype": "fp8", "m": 1, "n": 8, "k": 8}, "r")
        self.assertEqual((report.group_count, report.query_count, report.dropped_queries), (1, 2, 1))


class CollectiveFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.links = LinkCatalog(test_links())
        self.topology: TopologySpec = test_topology(node_size=4)
        self.layout = self.topology.group_layout([0, 1, 2, 3])
        self.ep_query = EPAllToAllQuery(self.layout, 32, 4096, 8, 256, "bf16", "deepep_high_throughput")

    def _model(self, fallback: str, report: MissingShapeReport | None = None, lookup=None) -> CollectiveModel:
        return CollectiveModel(
            self.topology, self.links, family="nvidia_gpu", fallback=fallback, missing_report=report, measured_lookup=lookup
        )

    def test_table_only_refuses_formula_pricing_of_communication(self):
        model = self._model("table_only")
        with self.assertRaises(MissingOperatorDataError) as ctx:
            model.estimate(CollectiveQuery("all_reduce", 1 << 20, self.layout, "bf16"))
        self.assertEqual((ctx.exception.table_name, ctx.exception.kind), ("collective", "table_absent"))
        with self.assertRaises(MissingOperatorDataError) as ctx:
            model.ep_all2all(self.ep_query)
        self.assertEqual((ctx.exception.table_name, ctx.exception.kind), ("ep_all2all", "table_absent"))
        with self.assertRaises(MissingOperatorDataError) as ctx:
            model.ep_all2all(EPAllToAllQuery(self.layout, 32, 4096, 8, 256, "bf16", "allgather_reducescatter"))
        self.assertEqual(ctx.exception.kind, "no_measured_mode")

    def test_table_first_prices_by_formula_and_reports_every_miss(self):
        report = MissingShapeReport()
        model = self._model("table_first", report)

        collective = model.estimate(CollectiveQuery("all_reduce", 1 << 20, self.layout, "bf16"))
        deepep = model.ep_all2all(self.ep_query)
        naive = model.ep_all2all(EPAllToAllQuery(self.layout, 32, 4096, 8, 256, "bf16", "naive"))

        self.assertEqual({collective.match_type, deepep.match_type, naive.match_type}, {"analytical"})
        groups = {(g["table"], g["key"].get("mode"), next(iter(g["kinds"]))) for g in report.records()}
        self.assertEqual(
            groups,
            {
                ("collective", None, "table_absent"),
                ("ep_all2all", "deepep_high_throughput", "table_absent"),
                ("ep_all2all", "naive", "no_measured_mode"),
            },
        )
        collective_group = next(g for g in report.records() if g["table"] == "collective")
        self.assertEqual(collective_group["key"], {"dtype": "bf16", "operation": "all_reduce", "group_size": 4, "nodes": 1})
        self.assertEqual(collective_group["axes"]["message_bytes"], [1 << 20, 1 << 20])

    def test_analytical_only_never_consults_or_reports(self):
        report = MissingShapeReport()
        source = SourceRecord("nccl:test", "A", "measured")
        meta = PackageMeta("v", "TestGPU", "nccl", sources={"nccl:test": source})
        rows = [
            {"dtype": "bf16", "operation": "all_reduce", "group_size": 4, "nodes": 1, "message_bytes": size,
             "latency_us": 1.0, "source_id": "nccl:test"}
            for size in (1 << 16, 1 << 24)
        ]
        package = OperatorDataPackage.from_rows(meta, {"collective": rows})
        model = self._model("analytical_only", report, OperatorLookup(package))

        estimate = model.estimate(CollectiveQuery("all_reduce", 1 << 20, self.layout, "bf16"))

        self.assertEqual(estimate.match_type, "analytical")
        self.assertGreater(estimate.latency_us, 1.0)
        self.assertEqual(report.group_count, 0)

    def test_measured_rows_leave_no_miss_behind(self):
        report = MissingShapeReport()
        source = SourceRecord("deepep:test", "A", "measured")
        meta = PackageMeta("v", "TestGPU", "vllm", sources={"deepep:test": source})
        rows = [
            {"dtype": "bf16", "phase": phase, "mode": "deepep_high_throughput", "ep_size": 4, "nodes": 1,
             "hidden_size": 4096, "top_k": 8, "num_experts": 256, "num_tokens": tokens, "latency_us": 10.0 * tokens,
             "source_id": "deepep:test"}
            for phase in ("dispatch", "combine")
            for tokens in (16, 64)
        ]
        package = OperatorDataPackage.from_rows(meta, {"ep_all2all": rows})
        model = self._model("table_first", report, OperatorLookup(package))

        self.assertEqual(model.ep_all2all(self.ep_query).match_type, "interpolated")
        self.assertEqual(report.group_count, 0)
        # a different node span is a discrete-key miss, recorded with ep_size/nodes
        wide = self.topology.group_layout(list(range(8)))
        model.ep_all2all(EPAllToAllQuery(wide, 32, 4096, 8, 256, "bf16", "deepep_high_throughput"))
        [group] = report.records()
        self.assertEqual((group["key"]["ep_size"], group["key"]["nodes"], group["kinds"]), (8, 2, {"discrete_key": 1}))


class EngineSharedReportTest(unittest.TestCase):
    def _run(self, fallback: str, tmpdir: str) -> tuple[dict, dict]:
        reset_g_time()
        env = simpy.Environment()
        device = test_device("TestGPU", memory_gib=0.05)
        model = test_model("TestModel", hidden_size=16, intermediate_size=32, num_layers=4, num_attention_heads=8)
        hardware: HardwareContext = test_hardware(device, models=[model])
        hardware.register_package(_gemm_only_package())
        config = ParallelConfig(tensor_parallel_size=2)
        cluster = ClusterConfig(
            num_workers=2,
            networks={"net1": "ethernet-test"},
            worker_groups=[WorkerGroupConfig(role="hybrid", hardware="TestGPU", num_workers=2, network="net1")],
        )
        psla = PSLAConfig(
            name="report-test", model="TestModel", distribution="burst", prefill_mean_len=16, prefill_range_len=0,
            decode_mean_len=1, decode_range_len=0, decode_len_distribution="uniform",
            first_token_latency=MetricData(0, 0, 0), decode_token_latency=MetricData(0, 0, 0), qps=1,
        )
        engine = LLMEngine(
            env=env, block_size=16, batching="paged-attn", kv_transfer_config=cluster.effective_kv_transfer(),
            psla_config=psla, cluster_config=cluster, parallel_config=config, hardware=hardware,
            prefill_worker_pool_type="round_robin", decode_worker_pool_type="round_robin", max_parallem_sum=8,
            max_occupy_ratio=1, latency_fallback=fallback, operator_backend="test",
        )
        requests = [Request(0, 16, 2, block_size=16)]
        env.process(LLMSource(env, engine, requests, qps=1, distribution="burst"))

        def stop_when_done():
            while env.now < 1:
                if requests[0].is_done or engine.failure is not None:
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-6)
                    return env.now
                yield env.timeout(1e-6)
            raise AssertionError("request did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)
        engine.raise_if_failed()
        stats = get_latency_stats(engine)
        args = argparse.Namespace(qps=1, batching="paged-attn", results_path=tmpdir, cluster="unused.json")
        export_result(
            args=args, g_time=g_time, engine=engine, model_config=psla, cluster=cluster, request_count=1,
            prefill_lens=[16], decode_lens=[2], requests=requests, notdone=[], duration=monitor.value,
        )
        return stats, json.loads((Path(tmpdir) / "missing_shapes_1.json").read_text())

    def test_worker_report_covers_compute_and_communication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats, report_file = self._run("table_first", tmpdir)

        tables = set(stats["operator_missing_per_table"])
        self.assertIn("collective", tables)  # TP all-reduce priced by formula
        self.assertIn("context_attention", tables)  # gemm-only package
        self.assertGreater(stats["operator_missing_shape_groups"], 0)
        self.assertEqual(stats["operator_table_match_counts"]["gemm"].get("analytical", 0), 0)
        shares = analytical_share_by_table(stats)
        self.assertEqual(shares.get("gemm", 0.0), 0.0)
        self.assertEqual(shares["context_attention"], 1.0)
        self.assertEqual(report_file["group_count"], stats["operator_missing_shape_groups"])
        self.assertEqual({g["table"] for g in report_file["groups"]}, tables)
        collective = next(g for g in report_file["groups"] if g["table"] == "collective")
        self.assertEqual(collective["key"]["group_size"], 2)
        self.assertEqual(collective["kinds"], {"table_absent": collective["query_count"]})

    def test_table_only_fails_the_run_on_the_first_miss(self):
        # The worker's exception stops the simulation and is re-raised by the
        # engine instead of exporting partial results.
        with tempfile.TemporaryDirectory() as tmpdir, self.assertLogs("TokenSim.llm.llm_engine", level="ERROR"), \
                self.assertRaises(SimulationStateError) as ctx:
            self._run("table_only", tmpdir)
        cause = ctx.exception.__cause__
        self.assertIsInstance(cause, MissingOperatorDataError)
        self.assertEqual(cause.kind, "table_absent")
        self.assertIn(cause.table_name, {"context_attention", "elementwise", "collective"})


@unittest.skipUnless((_B300_VLLM / "generation_meta.yaml").is_file(), "B300 vLLM package not present")
class B300AgentXShapesTest(unittest.TestCase):
    """The InferenceX B300 point: TP8/EP8 per replica on one HGX node, DeepSeek-V3 geometry."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.hardware = HardwareContext.load("data")
        cls.package = cls.hardware.operator_package("b300_sxm", "vllm")
        cls.lookup = OperatorLookup(cls.package)
        cls.topology = cls.hardware.topology("hgx_b200_8x_xdr")

    def test_b300_alias_loads_the_b300_sxm_vllm_package(self):
        device = self.hardware.device("B300")
        self.assertEqual(device.device_id, "b300_sxm")
        self.assertEqual((self.package.device_id, self.package.backend), ("b300_sxm", "vllm"))
        self.assertIn("ep_all2all", self.package.tables)

    def test_per_dp_ep8_queries_are_measured_and_wide_ep64_is_reported(self):
        report = MissingShapeReport()
        model = CollectiveModel(
            self.topology, self.hardware.links, family="nvidia_gpu", measured_lookup=self.lookup,
            fallback="table_first", missing_report=report,
        )
        node = self.topology.group_layout(list(range(8)))
        deepseek = dict(hidden_size=7168, top_k=8, num_experts=256, dtype="bf16")

        prefill = model.ep_all2all(EPAllToAllQuery(node, num_tokens=4096, mode="deepep_high_throughput", **deepseek))
        decode = model.ep_all2all(EPAllToAllQuery(node, num_tokens=64, mode="deepep_low_latency", **deepseek))
        allreduce = model.estimate(CollectiveQuery("all_reduce", 64 << 20, node, "bf16"))

        self.assertEqual(prefill.match_type, "measured")
        self.assertEqual((prefill.detail["mode"], prefill.detail["nodes"]), ("deepep_high_throughput", 1))
        self.assertEqual(decode.match_type, "measured")
        self.assertEqual(allreduce.match_type, "measured")
        self.assertEqual(report.group_count, 0)

        # the wide-EP layout of the old AgentX runs: 64 ranks over 8 nodes
        cluster = self.topology.group_layout(list(range(64)))
        wide = model.ep_all2all(EPAllToAllQuery(cluster, num_tokens=4096, mode="deepep_high_throughput", **deepseek))
        self.assertEqual(wide.match_type, "analytical")
        [group] = report.records()
        self.assertEqual((group["table"], group["key"]["ep_size"], group["key"]["nodes"]), ("ep_all2all", 64, 8))
        self.assertEqual(group["kinds"], {"discrete_key": 1})

    def test_agentic_context_lengths_extrapolate_attention_up_to_64x(self):
        for table, axis in (("context_attention", "input_seq_len"), ("generation_attention", "context_len")):
            spec = TABLE_SPECS[table]
            bound = next(a.max_extrapolation_ratio for a in spec.axes if a.name == axis)
            self.assertEqual(bound, ATTENTION_SEQ_EXTRAPOLATION_RATIO)
        self.assertEqual(ATTENTION_SEQ_EXTRAPOLATION_RATIO, 64.0)

        # DeepSeek-V3-like MLA geometry under TP8: 16 local heads, 1 kv head, head_dim 64
        model = ModelSpec(
            model_id="deepseek-like", hidden_size=7168, intermediate_size=18432, num_layers=61,
            num_attention_heads=128, num_key_value_heads=1, head_dim=64, dtype="bf16", kv_cache_dtype="bf16",
        )
        backend = OperatorTableLatencyBackend(
            device=self.hardware.device("B300"), model=model,
            parallel_config=ParallelConfig(tensor_parallel_size=8), package=self.package, fallback="table_first",
        )
        # measured prefill boundary 16384: 600k (37x) extrapolates, 1.2M (73x) is a reported miss
        backend._attention_prefill(1, 600_000, 600_000)
        self.assertEqual(backend.stats.table_match_counts["context_attention"], {"extrapolated": 1})
        backend._attention_prefill(1, 1_200_000, 1_200_000)
        self.assertEqual(backend.stats.table_match_counts["context_attention"]["analytical"], 1)
        # measured decode boundary 131072: a 4M-token context (31x) extrapolates
        backend._attention_decode(8, 4_000_000)
        self.assertEqual(backend.stats.table_match_counts["generation_attention"], {"extrapolated": 1})
        [miss] = backend.stats.missing_shape_records()
        self.assertEqual((miss["table"], miss["kinds"]), ("context_attention", {"out_of_range": 1}))
        self.assertEqual(miss["axes"]["input_seq_len"], [1_200_000, 1_200_000])


if __name__ == "__main__":
    unittest.main()
