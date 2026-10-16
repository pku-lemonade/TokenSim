"""Historical arrivals, user scheduling, exported evidence and runtime settings."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import simpy

from TokenSim.errors import SimulationStateError
from scripts.run_v3_user_qps import main, parse_args
from scripts.user_qps.configuration import historical_runtime
from scripts.user_qps.metrics import validate_and_summarize
from scripts.user_qps.runner import experiment, run_case
from scripts.user_qps.scheduler import schedule, workload


def synthetic_run(arrivals, *, users=2, batch=1, length=2, service=1):
    env = simpy.Environment()
    requests, tokens, batches, queues = [], [], [], []

    def serve(selected, iteration):
        yield env.timeout(service)
        return {request.id: env.now for request in selected}

    schedule(
        env,
        arrivals,
        users=users,
        max_batch=batch,
        output_tokens=length,
        context=2048,
        serve=serve,
        request_rows=requests,
        token_rows=tokens,
        batch_rows=batches,
        queue_rows=queues,
    )
    env.run()
    args = SimpleNamespace(
        users=users,
        max_batch=batch,
        output_tokens=length,
        context=2048,
        a_tp=4,
        f_ep=32,
        af_batch_size=8,
    )
    summary = validate_and_summarize(
        args, 4, arrivals, requests, tokens, batches, queues
    )
    return args, requests, tokens, batches, queues, summary


class UserQPSSchedulerTest(unittest.TestCase):
    def test_historical_poisson_seed_and_unique_marks(self):
        rows = workload(4, 32, 32, 42, unique_users=True)
        self.assertEqual(sorted(row["user"] for row in rows), list(range(32)))
        self.assertAlmostEqual(rows[0]["arrival_s"], 0.25501507181870024)
        self.assertAlmostEqual(rows[-1]["arrival_s"], 6.440222072339379)
        slow = workload(1, 32, 32, 42, unique_users=True)
        self.assertEqual([r["user"] for r in rows], [r["user"] for r in slow])
        self.assertEqual(
            [4 * r["arrival_s"] for r in rows], [r["arrival_s"] for r in slow]
        )
        with self.assertRaises(ValueError):
            workload(1, 3, 2, 42, unique_users=True)

    def test_arrivals_during_compute_keep_fifo_and_round_robin(self):
        arrivals = [
            dict(request_id=0, user=0, arrival_s=0.1),
            dict(request_id=1, user=0, arrival_s=0.2),
            dict(request_id=2, user=1, arrival_s=0.3),
        ]
        _, requests, tokens, _, queues, summary = synthetic_run(arrivals)
        self.assertEqual([t["request_id"] for t in tokens], [0, 2, 0, 2, 1, 1])
        self.assertEqual(
            [t["context_len"] for t in tokens], [2048, 2048, 2049, 2049, 2048, 2049]
        )
        self.assertAlmostEqual(requests[0]["completion_s"], 3.1)
        self.assertAlmostEqual(requests[1]["first_admitted_s"], 4.1)
        self.assertAlmostEqual(summary["total_tps"], 6 / 6.1)
        self.assertEqual(summary["max_pending"], 1)
        self.assertEqual(queues[-1]["active_users"], 0)

    def test_full_32_by_400_accounting_and_latency_identity(self):
        arrivals = workload(4, 32, 32, 42, unique_users=True)
        _, requests, tokens, batches, queues, summary = synthetic_run(
            arrivals, users=32, batch=8, length=400, service=0.04
        )
        self.assertEqual(summary["completed_requests"], 32)
        self.assertEqual(summary["total_output_tokens"], 12800)
        self.assertEqual(sum(t["itl_ms"] is not None for t in tokens), 32 * 399)
        self.assertEqual(summary["max_actual_batch"], 8)
        self.assertEqual(summary["dropped_requests"], 0)
        self.assertGreater(summary["drain_s"], 0)
        self.assertEqual(queues[-1]["runnable"], 0)
        self.assertEqual(batches[0]["batch_size"], 1)
        for request in requests:
            selected = [t for t in tokens if t["request_id"] == request["request_id"]]
            ttft = selected[0]["completion_s"] - request["arrival_s"]
            intervals = sum(t["itl_ms"] for t in selected[1:]) / 1000
            self.assertAlmostEqual(
                ttft + intervals, request["completion_s"] - request["arrival_s"]
            )
        self.assertAlmostEqual(
            summary["mean_latency_ms"],
            summary["mean_ttft_ms"] + 399 * summary["mean_tbt_ms"],
        )
        self.assertGreaterEqual(summary["max_latency_ms"], summary["p95_latency_ms"])

    def test_one_token_has_ttft_and_latency_but_no_tbt(self):
        result = synthetic_run([dict(request_id=0, user=0, arrival_s=0.1)], length=1)
        summary = result[-1]
        self.assertIsNone(summary["mean_tbt_ms"])
        self.assertIsNone(summary["max_tbt_ms"])
        self.assertEqual(summary["mean_ttft_ms"], summary["mean_latency_ms"])

    def test_audit_rejects_corrupted_token_and_nonempty_final_queue(self):
        arrivals = [dict(request_id=0, user=0, arrival_s=0.1)]
        args, requests, tokens, batches, queues, _ = synthetic_run(arrivals)
        bad_tokens = deepcopy(tokens)
        bad_tokens[-1]["itl_ms"] += 100
        with self.assertRaisesRegex(SimulationStateError, "TBT"):
            validate_and_summarize(
                args, 4, arrivals, requests, bad_tokens, batches, queues
            )
        bad_queues = deepcopy(queues)
        bad_queues[-1]["pending"] = 1
        with self.assertRaisesRegex(SimulationStateError, "final user queues"):
            validate_and_summarize(
                args, 4, arrivals, requests, tokens, batches, bad_queues
            )

    def test_parser_defaults_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "new")
            defaults = parse_args(["--output", output])
            self.assertEqual(
                (defaults.qps, defaults.users, defaults.requests), ([4], 32, 32)
            )
            self.assertEqual((defaults.context, defaults.output_tokens), (2048, 400))
            self.assertTrue(defaults.unique_users)
            for extra in (
                ["--qps", "nan"],
                ["--qps", "0"],
                ["--requests", "33"],
                ["--output-tokens", "0"],
                ["--qps", "4", "4"],
                ["--a-tp", "1"],
            ):
                with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(["--output", output, *extra])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--output", directory])


class UserQPSRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = parse_args(["--output", "outputs/unit-test-unused"])
        cls.runtime = historical_runtime(cls.args)

    def test_full_model_physical_placement_and_explicit_domains(self):
        runtime = self.runtime
        self.assertEqual(runtime.plan.total_layers, 61)
        af = runtime.cluster.af
        domains = af.communication_domains
        self.assertFalse(af.communication_timing.enabled)
        self.assertEqual((domains.node_size, domains.node_count), (8, 5))
        self.assertEqual(domains.placement_strategy, "dedicated")
        self.assertEqual(domains.local_bandwidth_bytes_per_s, 300e9)
        self.assertEqual(domains.inter_tx_bandwidth_bytes_per_s, 50e9)
        self.assertEqual(domains.inter_rx_bandwidth_bytes_per_s, 50e9)
        self.assertEqual((domains.local_latency_us, domains.inter_latency_us), (0, 0))
        self.assertEqual(
            sorted(runtime.placement.device_index_by_worker.values()),
            list(range(4)) + list(range(8, 40)),
        )
        a, f = runtime.plan.domains
        self.assertEqual(
            (a.tensor_parallel_size, f.tensor_parallel_size, f.expert_parallel_size),
            (4, 32, 32),
        )
        self.assertEqual(f.expert_tensor_parallel_size, 1)
        for state in runtime.plan.physical_states:
            device = runtime.hardware.device(state.hardware)
            self.assertEqual(
                device.memory_capacity_bytes.value,
                (
                    1536 * 2**30
                    if state.physical_identity.startswith("f-pool")
                    else runtime.hardware.device(
                        "a100_sxm_80g"
                    ).memory_capacity_bytes.value
                ),
            )
            self.assertEqual(state.capacity_bytes, int(device.usable_memory_bytes))

    def test_provenance_covers_new_helpers_and_current_runtime(self):
        record = experiment(self.args, self.runtime)
        for name in (
            "scripts/run_v3_user_qps.py",
            "scripts/user_qps/scheduler.py",
            "TokenSim/af/execution/model.py",
        ):
            self.assertEqual(len(record["source_sha256"][name]), 64)
        self.assertEqual(record["phase"], "decode_only")

    def test_dry_run_exports_default_arrivals_without_success_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dry"
            with (
                patch(
                    "scripts.run_v3_user_qps.historical_runtime",
                    return_value=self.runtime,
                ),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(["--output", str(output), "--dry-run"]), 0)
            point = output / "qps-4"
            self.assertEqual(len(json.loads((point / "arrivals.json").read_text())), 32)
            self.assertEqual(
                json.loads((point / "progress.json").read_text())["status"], "planned"
            )
            self.assertFalse((point / "summary.json").exists())
            self.assertFalse((output / "results.json").exists())

    def test_model_failure_retains_evidence_without_completed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(
                [
                    "--output",
                    str(Path(directory) / "failure"),
                    "--requests",
                    "1",
                    "--output-tokens",
                    "1",
                ]
            )
            args.output.mkdir()

            def fail(executor, iteration):
                raise RuntimeError("injected execution failure")
                yield

            with (
                patch("scripts.user_qps.runner.ModelAFExecutor.run_iteration", fail),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    run_case(args, self.runtime, 4)
            point = args.output / "qps-4"
            self.assertTrue((point / "failure.json").exists())
            self.assertTrue((point / "requests.json").exists())
            self.assertFalse((point / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
