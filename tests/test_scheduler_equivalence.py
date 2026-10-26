"""Queue optimization preserves complete AgentX results and replay evidence."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from TokenSim.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]


class ListQueue(list):
    def popleft(self):
        return self.pop(0)


class SchedulerEquivalenceTest(unittest.TestCase):
    def test_agentx_results_and_all_replay_logs_match_list_queue(self):
        with TemporaryDirectory() as directory:

            def run(legacy):
                args = build_parser().parse_args(
                    [
                        "--model",
                        str(ROOT / "data/models/llama-7b.yaml"),
                        "--cluster",
                        str(ROOT / "data/clusters/1_a100/h1.json"),
                        "--workload_type",
                        "agentx_weka",
                        "--dataset_path",
                        str(ROOT / "tests/fixtures/agentx_weka.jsonl"),
                        "--batching",
                        "paged-attn",
                        "--block_size",
                        "64",
                        "--latency_backend",
                        "analytical",
                        "--max_num_seqs",
                        "4",
                        "--max_num_batched_tokens",
                        "64",
                        "--agentx_concurrency",
                        "2",
                        "--agentx_profile_duration",
                        "8",
                        "--no-agentx_warmup",
                        "--verbose",
                        "none",
                        "--results_path",
                        directory,
                    ]
                )
                with redirect_stdout(io.StringIO()):
                    if legacy:
                        with patch("TokenSim.llm.paged_scheduling.deque", ListQueue):
                            main(args)
                    else:
                        main(args)
                root = Path(directory)
                result = json.loads((root / "agentx_c2.json").read_text())
                result.pop("simulator_wall_time")
                logs = {p.name: p.read_bytes() for p in (root / "replay").iterdir()}
                self.assertEqual(result["termination_reason"], "completed")
                self.assertEqual(result["notdone"], [])
                self.assertTrue(logs["requests.jsonl"])
                return result, logs

            self.assertEqual(run(True), run(False))
