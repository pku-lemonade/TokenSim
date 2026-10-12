"""Human-readable simulation statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing import List, Any
from TokenSim.results import MetricData
from TokenSim.llm.llm_request import LLMTime, Request
from .stats import (
    get_prefix_reuse_stats,
    get_connector_stats,
    get_parallel_stats,
    get_latency_stats,
    analytical_share_by_table,
    get_moe_stats,
    get_mooncake_stats,
)

if TYPE_CHECKING:
    from TokenSim.llm.llm_engine import LLMEngine


def print_all_stats(
    g_time: LLMTime,
    requests: list[Request],
    engine: LLMEngine,
    duration: float,
    request_count: int,
    prefill_lens: list[int],
    slo: dict[str, Any] | None = None,
):
    request_time = MetricData.from_list(g_time.request_time)
    prefill_time = MetricData.from_list(g_time.prefill_time)
    decode_time = MetricData.from_list(g_time.decode_time)
    decode_max_time = MetricData.from_list(g_time.decode_max_time)
    prefill_idle_time = MetricData.from_list(g_time.prefill_idle)
    decode_idle_time = MetricData.from_list(g_time.decode_idle)

    # Print timing metrics.
    print_metrics(
        request_time,
        prefill_time,
        decode_time,
        decode_max_time,
        prefill_idle_time,
        decode_idle_time,
    )
    prefill_latency = [req.prefill_latency for req in requests]
    decode_latency = [
        req.decode_time_sum / max(1, req.generation_idx - 1) for req in requests
    ]
    # Print latency statistics.
    print_latency_stats(prefill_latency, decode_latency)
    # Print throughput statistics.
    print_throughput_stats(
        duration, request_count, [r.generation_idx for r in requests]
    )
    print(f"Total preemptions: {sum([wkr.preempted_cnt for wkr in engine.workers])}")
    print_prefix_reuse_stats(requests)
    print_connector_stats(engine)
    print_mooncake_stats(engine)
    print_parallel_stats(engine)
    print_moe_stats(engine)
    print_operator_latency_stats(engine)
    print_slo_stats(slo)


def print_metrics(
    request_time: MetricData,
    prefill_time: MetricData,
    decode_time: MetricData,
    decode_max_time: MetricData,
    prefill_idle_time: MetricData,
    decode_idle_time: MetricData,
):
    """Print aggregate timing metrics.

    Args:
        request_time: End-to-end request timing metric.
        prefill_time: Prefill processing timing metric.
        decode_time: Decode processing timing metric.
        decode_max_time: Maximum decode timing metric.
        prefill_idle_time: Prefill worker idle-time metric.
        decode_idle_time: Decode worker idle-time metric.
    """
    print(f"{request_time=}")
    print(f"{prefill_time=}")
    print(f"{decode_time=}")
    print(f"{decode_max_time=}")
    print(f"{prefill_idle_time=}")
    print(f"{decode_idle_time=}")


def print_latency_stats(prefill_latency: List[float], decode_latency: List[float]):
    print(f"Average prefill latency: {sum(prefill_latency) / len(prefill_latency)}")
    print(f"Max prefill latency: {max(prefill_latency)}")
    print(f"Min prefill latency: {min(prefill_latency)}")
    print(f"Average decode latency: {sum(decode_latency) / len(decode_latency)}")
    print(f"Max decode latency: {max(decode_latency)}")
    print(f"Min decode latency: {min(decode_latency)}")


def print_throughput_stats(dur: float, request_count: int, output_tokens: List[int]):
    print(f"total time: {dur}")
    print(
        f"Throughput: {request_count / dur} r/s, {sum(output_tokens) / dur} output token/s"
    )


def print_prefix_reuse_stats(requests: list[Request]):
    stats = get_prefix_reuse_stats(requests)
    print(
        "Prefix cache: "
        + f"hit_blocks={stats['reuse_hit_blocks']}, "
        + f"miss_blocks={stats['reuse_miss_blocks']}, "
        + f"hit_tokens={stats['reuse_hit_tokens']}, "
        + f"effective_prefill_tokens={stats['effective_prefill_tokens']}, "
        + f"hit_rate={stats['prefix_cache_hit_rate']}"
    )


def print_connector_stats(engine: LLMEngine):
    stats = get_connector_stats(engine)
    if not stats:
        return
    print(
        "KV connector: "
        + f"transfers={stats['connector_transfer_count']}, "
        + f"blocks={stats['connector_transfer_blocks']}, "
        + f"bytes={stats['connector_transfer_bytes']}, "
        + f"latency={stats['connector_transfer_latency']}, "
        + f"load_wait={stats['connector_load_wait_time']}, "
        + f"save_wait={stats['connector_save_wait_time']}"
    )


def print_mooncake_stats(engine: LLMEngine):
    stats = get_mooncake_stats(engine)
    if not (
        stats.get("mooncake_get_count")
        or stats.get("mooncake_put_count")
        or stats.get("mooncake_transferred_bytes")
    ):
        return
    print(
        "Mooncake: "
        + f"gets={stats['mooncake_get_count']}, "
        + f"puts={stats['mooncake_put_count']}, "
        + f"hits={stats['mooncake_store_hit_count']}, "
        + f"misses={stats['mooncake_store_miss_count']}, "
        + f"mem_hits={stats['mooncake_memory_tier_hit_count']}, "
        + f"disk_hits={stats['mooncake_disk_tier_hit_count']}, "
        + f"ssd_read_blocks={stats['mooncake_ssd_read_blocks']}, "
        + f"ssd_write_blocks={stats['mooncake_ssd_write_blocks']}, "
        + f"bytes={stats['mooncake_transferred_bytes']}, "
        + f"latency={stats['mooncake_transfer_latency']}"
    )


def print_parallel_stats(engine: LLMEngine):
    stats = get_parallel_stats(engine)
    config = stats.get("parallel_config", {})
    groups = stats.get("parallel_groups", {})
    print(
        "Parallel: "
        + f"config={config}, "
        + f"ranks={stats.get('parallel_actual_rank_count')}/"
        + f"{stats.get('parallel_expected_rank_count')}, "
        + f"dp_placements={stats.get('parallel_dp_placement_counts')}, "
        + f"tp_group={groups.get('tp_group_fan')}@{groups.get('tp_group_link')}, "
        + f"ep_group={groups.get('ep_group_size')}x{groups.get('ep_group_count')}@{groups.get('ep_group_link')}, "
        + f"sync_events={stats.get('parallel_sync_event_count', 0)}, "
        + f"sync_latency={stats.get('parallel_latency_total', 0)}"
    )


def print_moe_stats(engine: LLMEngine):
    stats = get_moe_stats(engine)
    config = stats.get("effective_moe_config", {})
    if not config.get("is_moe_model", False):
        print("MoE: disabled")
        return
    all2all = get_parallel_stats(engine).get("parallel_ep_all2all_latency", 0.0)
    print(
        "MoE: "
        + f"experts={config.get('num_experts')}, "
        + f"top_k={config.get('num_experts_per_tok')}, "
        + f"ep_ranks={stats.get('moe_ep_rank_count')}, "
        + f"imbalance={stats.get('moe_expert_load_imbalance_ratio')}, "
        + f"compute_latency={stats.get('moe_compute_latency')}, "
        + f"all2all_latency={all2all}, "
        + f"straggler_latency={stats.get('moe_straggler_latency')}"
    )


def print_operator_latency_stats(engine: LLMEngine):
    stats = get_latency_stats(engine)
    stats.update(get_parallel_stats(engine))
    backends = stats.get("latency_backends", {})
    print(
        "Latency: "
        + f"backends={ {k: v.get('backend') + ':' + str(v.get('operator_backend')) for k, v in backends.items()} }, "
        + f"queries={stats.get('operator_query_count', 0)}, "
        + f"match_types={stats.get('operator_match_type_counts', {})}, "
        + f"missing_shapes={stats.get('operator_missing_shape_count', 0)} "
        + f"in {stats.get('operator_missing_shape_groups', 0)} groups"
    )
    formula_share = {
        table: f"{share:.0%}"
        for table, share in analytical_share_by_table(stats).items()
        if share > 0
    }
    if formula_share:
        # Loud on purpose: any share here is simulated time resting on formulas
        # rather than measurements, see missing_shapes_<qps>.json.
        print(f"Latency WARNING analytical share per table: {formula_share}")
    for group in stats.get("operator_missing_shapes", [])[:5]:
        print(
            f"  missing {group['table']} {group['key']} axes={group['axes']} "
            + f"queries={group['query_count']} kinds={group['kinds']}"
        )
    components = stats.get("operator_component_seconds", {})
    if components:
        print(
            "Latency breakdown (s): "
            + ", ".join(f"{name}={value:.4f}" for name, value in components.items())
        )


def print_slo_stats(report: dict[str, Any] | None) -> None:
    if not report:
        return
    for name in ("first_token_latency", "decode_token_latency"):
        if name in report:
            entry = report[name]
            print(
                f"SLO {name}: "
                + ", ".join(
                    f"{key} {entry['observed'][key]} <= {bound}: {entry['met'][key]}"
                    for key, bound in entry["target"].items()
                )
            )
    if "goodput_requests" in report:
        print(
            f"SLO goodput: {report['goodput_requests']} r, "
            f"{report['goodput_requests_per_s']} r/s"
        )
