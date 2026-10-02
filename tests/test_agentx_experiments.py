"""Validation tests for the unified AgentX experiment matrix entrypoint."""

import json
from pathlib import Path

from scripts.run_agentx_experiments import check, load_matrix, run_point, select, summarize


def test_matrix_loads_and_resolves_checkpoint_defaults():
    document, points = load_matrix(Path("data/experiments/agentx.yaml"))
    assert {point["id"] for point in select(points, group="checkpoints")} == {
        "b200-c196-dep8-simplecpu",
        "b300-c384-dep8-simplecpu",
    }
    assert all(point["trace_count"] == 393 for point in points)
    assert all(point["profiling_window_s"] == 3600 for point in points)
    assert check(points)["valid"]
    assert document["reference_curve"]["status"] == "reference-index-only"
    assert len(document["reference_curve"]["points"]) == 21


def test_matrix_selection_and_summary(tmp_path):
    _, points = load_matrix(Path("data/experiments/agentx.yaml"))
    assert [point["hardware"] for point in select(points, hardware="b200", group="checkpoints")] == ["b200"]
    result = tmp_path / "b200" / "agentx_c196.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "agentx_metrics": {
                    "output_token_throughput_tps": 1,
                    "input_token_throughput_tps": 2,
                    "ttft_s": {"p90": 3},
                    "tpot_s": {"p90": 4},
                    "gpu_cache_hit_rate": 0.5,
                    "cpu_cache_hit_rate": 0.2,
                    "request_count": 5,
                }
            }
        )
    )
    output = tmp_path / "summary.csv"
    assert summarize(tmp_path, output)["rows"] == 1
    assert "output_token_throughput_tps" in output.read_text()


def test_summary_records_actual_gpu_normalization_and_seed_aggregate(tmp_path):
    for seed, value in ((1, 8.0), (2, 10.0)):
        result = tmp_path / f"seed{seed}" / "b200-c196-dep8-simplecpu" / "b200" / "agentx_c196.json"
        result.parent.mkdir(parents=True)
        (result.parent / "run_manifest.json").write_text(
            json.dumps(
                {
                    "simulation_seed": seed,
                    "point": {
                        "id": "b200-c196-dep8-simplecpu",
                        "num_gpus": 8,
                        "cache_backend": "simple_cpu",
                        "parallelism": {"tp": 1, "dp": 8, "ep": 8},
                    },
                }
            )
        )
        result.write_text(
            json.dumps(
                {
                    "agentx_metrics": {
                        "output_token_throughput_tps": value,
                        "input_token_throughput_tps": 16,
                        "ttft_s": {"p90": 3},
                        "tpot_s": {"p90": 4},
                        "e2e_s": {"p90": 5},
                        "gpu_cache_hit_rate": 0.5,
                        "cpu_cache_hit_rate": 0.2,
                        "request_count": 5,
                    }
                }
            )
        )
    output = tmp_path / "summary.csv"
    summarize(tmp_path, output)
    rows = list(__import__("csv").DictReader(output.open()))
    aggregate = next(row for row in rows if row["row_type"] == "aggregate")
    assert float(aggregate["output_token_throughput_per_gpu_tps"]) == 1.125
    assert float(aggregate["output_tps_stdev"]) == 1.0


def test_successful_point_is_skipped_for_resume(tmp_path):
    target = tmp_path / "seed0" / "point" / "b200"
    target.mkdir(parents=True)
    (target / "run_status.json").write_text(json.dumps({"exit_code": 0}))
    point = {"id": "point", "hardware": "b200"}
    result = run_point(point, tmp_path, "simple_cpu")
    assert result == {"id": "point", "seed": 0, "exit_code": 0, "skipped": True}
