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


def test_successful_point_is_skipped_for_resume(tmp_path):
    target = tmp_path / "seed0" / "point" / "b200"
    target.mkdir(parents=True)
    (target / "run_status.json").write_text(json.dumps({"exit_code": 0}))
    point = {"id": "point", "hardware": "b200"}
    result = run_point(point, tmp_path, "simple_cpu")
    assert result == {"id": "point", "exit_code": 0, "skipped": True}
