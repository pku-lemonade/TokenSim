"""Simulation results, statistics and exports."""

from .stats import (
    get_prefix_reuse_stats,
    get_connector_stats,
    get_parallel_stats,
    get_cache_capacity_stats,
    get_latency_stats,
    _merge_sum,
    _add_module_stats,
    _finish_module_stats,
    analytical_share_by_table,
    get_moe_stats,
    get_mooncake_stats,
)
from .metrics import (
    _is_root_stream,
    RunReport,
    get_agentx_metrics,
    _percentile_summary,
    slo_report,
)
from .console import (
    print_all_stats,
    print_metrics,
    print_latency_stats,
    print_throughput_stats,
    print_prefix_reuse_stats,
    print_connector_stats,
    print_mooncake_stats,
    print_parallel_stats,
    print_moe_stats,
    print_operator_latency_stats,
    print_slo_stats,
)
from .export import export_result, _missing_shape_report, write_failure
