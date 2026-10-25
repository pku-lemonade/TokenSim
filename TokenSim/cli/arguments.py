"""Command-line arguments for workload simulation."""

from __future__ import annotations

import argparse
from TokenSim.config.parallel_config import EXPERT_PARALLEL_SCOPES
from TokenSim.config.paths import default_data_root
from TokenSim.latency import FALLBACK_POLICIES
from TokenSim.llm.llm_scheduler import DEFAULT_MAX_NUM_BATCHED_TOKENS


DATA_ROOT = default_data_root()
LATENCY_BACKEND_KEYWORDS = ("operator_table", "analytical")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace_mode",
        choices=("off", "validate", "full"),
        default=None,
        help="Global operator recording mode (default: off).",
    )
    parser.add_argument(
        "--trace_output", default=None, help="Global operator trace output path."
    )
    parser.add_argument(
        "--trace_config", default=None, help="JSON global trace configuration."
    )
    parser.add_argument(
        "--sim_time",
        type=float,
        default=None,
        help="Stop at this simulated time (seconds) and export partial results.",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=None,
        help="Open-loop request arrival rate; overrides the workload config.",
    )
    parser.add_argument(
        "--batching",
        choices=["static", "dynamic", "paged-attn"],
        default="dynamic",
        help="static/dynamic batch whole requests; paged-attn runs continuous batching over paged KV blocks.",
    )
    parser.add_argument(
        "--distribution",
        choices=["burst", "uniform", "poisson"],
        default=None,
        help="Arrival process: uniform spacing, Poisson, or a burst at time zero.",
    )
    parser.add_argument(
        "--request_count",
        type=int,
        default=100,
        help="Number of synthetic requests, or the maximum loaded from a dataset.",
    )
    parser.add_argument(
        "--prefill_mean_len", type=int, help="Mean synthetic prompt length in tokens."
    )
    parser.add_argument(
        "--prefill_range_len",
        type=int,
        help="Width of the uniform prompt-length range around the mean.",
    )
    parser.add_argument(
        "--decode_mean_len", type=int, help="Mean synthetic output length in tokens."
    )
    parser.add_argument(
        "--decode_range_len",
        type=int,
        help="Width of the output-length range around the mean.",
    )
    parser.add_argument(
        "--decode_len_distribution",
        choices=["uniform", "exponential", "capped_exponential", "burst"],
        default=None,
        help="Distribution of synthetic output lengths.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Tokens per KV-cache block (the logical block for grouped caches).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LLaMa-7B",
        help="Model catalog name or model YAML path.",
    )
    parser.add_argument(
        "--workload_config",
        default=None,
        help="Optional JSON request-generation and SLO configuration.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=str(DATA_ROOT / "clusters/1_a100/h1.json"),
        help="Cluster JSON: workers, hardware, topology, parallelism and storage.",
    )
    parser.add_argument(
        "--kv_transfer_config",
        type=str,
        default=None,
        help="KV connector JSON; replaces the cluster's storage configuration.",
    )
    parser.add_argument(
        "--prefill_worker_pool_type",
        choices=["round_robin", "least_gpu_memory", "balanced_load"],
        default="round_robin",
        help="How requests are assigned to prefill (or hybrid) workers.",
    )
    parser.add_argument(
        "--decode_worker_pool_type",
        choices=["round_robin", "least_gpu_memory", "balanced_load"],
        default="least_gpu_memory",
        help="How prefilled requests are assigned to decode workers.",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=64,
        help="Maximum running sequences per worker; also sizes CUDA-graph capture buffers.",
    )
    parser.add_argument(
        "--max_occupy_ratio",
        type=float,
        default=1.0,
        help="Stop admitting new requests above this fraction of GPU KV blocks in use.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=1.0,
        help="Fraction of device memory available to weights, activations and KV cache.",
    )
    parser.add_argument(
        "--router_policy",
        choices=["none", "consistent_hash", "fixed_mapping"],
        default="none",
        help="Request routing across DP ranks: worker-pool policy, consistent session hashing or a fixed mapping.",
    )
    parser.add_argument(
        "--router_backend_url",
        default="http://localhost:8889",
        help="Backend URL hashed into consistent-hash ring keys, as in vllm-router.",
    )
    parser.add_argument(
        "--router_mapping_path",
        default=None,
        help="JSON mapping of session IDs to ranks for --router_policy fixed_mapping.",
    )
    parser.add_argument(
        "--dp_lockstep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run data-parallel ranks in synchronized steps (dummy batches on idle ranks).",
    )
    parser.add_argument(
        "--dp_sync_interval",
        type=int,
        default=16,
        help="Steps between finished-state consensus checks under DP lockstep.",
    )
    parser.add_argument(
        "--long_prefill_token_threshold",
        type=int,
        default=0,
        help="Per-step chunk limit for long prompts while other requests wait; 0 disables.",
    )
    parser.add_argument(
        "--prefill_schedule_interval",
        type=int,
        default=1,
        help="With data parallelism, schedule prefill chunks only every N synchronized steps.",
    )
    parser.add_argument(
        "--adaptive_long_prefill_threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise the long-prefill limit to the step budget divided by eligible requests.",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        help="Token budget of one paged-attn step (vLLM V1 chunked prefill): decodes are served first, prompts longer than the remaining budget are prefilled in chunks.",
    )
    parser.add_argument(
        "--chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable to prefill every prompt in a single step regardless of length.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Override the cluster's tensor-parallel size.",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=None,
        help="Override the cluster's pipeline-parallel size.",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=None,
        help="Override the cluster's data-parallel size.",
    )
    parser.add_argument(
        "--data_parallel_rank",
        type=int,
        default=None,
        help="Recorded DP rank of this deployment (validated metadata; all ranks are simulated).",
    )
    parser.add_argument(
        "--data_parallel_size_local",
        type=int,
        default=None,
        help="DP ranks per node (validated and reported as topology metadata).",
    )
    parser.add_argument(
        "--enable_expert_parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Shard MoE experts across ranks instead of tensor-parallel experts.",
    )
    parser.add_argument(
        "--expert_parallel_scope",
        choices=list(EXPERT_PARALLEL_SCOPES),
        default=None,
        help="Which ranks share one copy of the experts: every TP x DP rank (global) or one DP replica (per_dp).",
    )
    parser.add_argument(
        "--expert_parallel_size",
        type=int,
        default=None,
        help="Ranks per expert-parallel group; validated against the TP/DP layout and the scope.",
    )
    parser.add_argument(
        "--expert_placement_strategy",
        choices=["linear", "round_robin"],
        default=None,
        help="Expert-to-rank assignment: contiguous blocks or round-robin.",
    )
    parser.add_argument(
        "--all2all_backend",
        choices=[
            "allgather_reducescatter",
            "naive",
            "deepep_high_throughput",
            "deepep_low_latency",
        ],
        default=None,
        help="Expert-parallel dispatch/combine implementation used for communication cost.",
    )
    parser.add_argument(
        "--moe_routing_distribution",
        choices=["uniform", "skew", "hot", "burst"],
        default=None,
        help="Synthetic expert routing when requests carry no expert histograms.",
    )
    parser.add_argument(
        "--moe_hot_experts",
        type=str,
        default=None,
        help="Comma-separated expert IDs for skewed synthetic routing.",
    )
    parser.add_argument(
        "--moe_hot_expert_fraction",
        type=float,
        default=None,
        help="Fraction of routed tokens sent to the hot experts.",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        choices=["none", "simple", "tqdm"],
        default="tqdm",
        help="Progress output: none, simple lines or a tqdm bar.",
    )
    parser.add_argument(
        "--debug_print",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print each request completion with simulated and wall-clock time.",
    )
    parser.add_argument(
        "--program_id",
        type=int,
        default=0,
        help="Label printed with simple progress lines when several runs share a terminal.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="",
        help="Result directory; defaults to results/<model>/<cluster directory>/<cluster>.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Request dataset or AgentX trace file.",
    )
    parser.add_argument(
        "--dataset_skip_count",
        type=int,
        default=0,
        help="Skip this many leading dataset records or traces.",
    )
    parser.add_argument(
        "--trace_timestamp_scale",
        type=float,
        default=None,
        help="Scale loaded timestamped trace arrivals as new_ts = old_ts * scale.",
    )
    parser.add_argument(
        "--trace_target_qps",
        type=float,
        default=None,
        help="Scale timestamped trace arrivals to this QPS using scale = original_trace_qps / trace_target_qps.",
    )
    parser.add_argument(
        "--workload_type",
        choices=["synthetic", "json_pairs", "qwen_jsonl", "agentx_weka"],
        default="synthetic",
        help="Request source: synthetic lengths, a dataset format, or AgentX trace replay.",
    )
    parser.add_argument(
        "--agentx_concurrency",
        type=int,
        default=1,
        help="Concurrent AgentX trace trees (replay lanes).",
    )
    parser.add_argument(
        "--agentx_trace_count",
        type=int,
        default=None,
        help="Use at most this many traces from the dataset; default all.",
    )
    parser.add_argument(
        "--agentx_profile_duration",
        type=float,
        default=3600.0,
        help="Simulated seconds of the measured AgentX issue window.",
    )
    parser.add_argument(
        "--agentx_issue_mode",
        choices=("closed_loop", "recorded"),
        default="closed_loop",
        help="Issue requests when their dependencies finish (closed loop) or at recorded trace times.",
    )
    parser.add_argument(
        "--agentx_idle_gap_cap",
        type=float,
        default=10.0,
        help="Compress system-wide idle gaps longer than this many seconds; 0 disables.",
    )
    parser.add_argument(
        "--agentx_warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm caches with sampled trace frontiers before the measured window.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Seed for synthetic workloads, routing and speculative acceptance.",
    )
    parser.add_argument(
        "--latency_backend",
        type=str,
        default="operator_table",
        help="'operator_table' (per-operator tables with analytical fallback) or 'analytical' (formulas only).",
        choices=list(LATENCY_BACKEND_KEYWORDS),
    )
    parser.add_argument(
        "--latency_fallback",
        choices=list(FALLBACK_POLICIES),
        default="table_first",
        help="How operator_table handles shapes missing from the tables.",
    )
    parser.add_argument(
        "--operator_backend",
        type=str,
        default=None,
        help="Operator-data backend for workers whose cluster group does not name one.",
        choices=["vllm", "sglang", "trtllm", "analytical"],
    )
    parser.add_argument(
        "--moe_backend",
        choices=("generic", "deep_gemm_amxf4_mega_moe"),
        default="generic",
        help="MoE cost source: generic operator tables or the fused DeepGEMM MegaMoE module binding.",
    )
    parser.add_argument(
        "--disable_moe_shared_fusion",
        action="store_true",
        help="Price shared experts separately instead of fused into the MoE module.",
    )
    parser.add_argument(
        "--attention_backend",
        choices=("generic", "module"),
        default="generic",
        help="Attention cost source: generic operators or the model's measured module binding.",
    )
    parser.add_argument(
        "--mtp_acceptance_length",
        type=float,
        default=1.0,
        help="Average accepted output tokens per speculative verification step.",
    )
    parser.add_argument(
        "--mtp_num_speculative_tokens",
        type=int,
        default=None,
        help="Draft tokens K per speculative step; 0 (the default) disables drafting.",
    )
    parser.add_argument(
        "--mhc_backend",
        choices=("generic", "module"),
        default="generic",
        help="Residual-mixing cost source: generic operators or the model's measured module binding.",
    )
    parser.add_argument(
        "--decode_context_bucket",
        type=int,
        default=128,
        help="Decode attention lookups round context length up to this multiple.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DATA_ROOT),
        help="Directory holding devices/, topologies/, models/ and operator_data/.",
    )
    parser.add_argument(
        "--agentx_workload_seed",
        type=int,
        default=None,
        help="Trace warmup sampling seed; defaults to random_seed.",
    )
    parser.add_argument(
        "--agentx_grace_period",
        type=float,
        default=30.0,
        help="Seconds to allow issued requests to finish.",
    )
    parser.add_argument(
        "--agentx_warmup_requests_per_lane",
        type=int,
        default=10,
        help="Warmup requests issued per replay lane before the measured window.",
    )
    parser.add_argument(
        "--agentx_warmup_grace_period",
        type=float,
        default=1800.0,
        help="Seconds allowed for warmup requests to finish before measurement.",
    )
    parser.add_argument(
        "--agentx_tree_idle_cap",
        type=float,
        default=300.0,
        help="Skip idle gaps longer than this within one trace tree; 0 disables.",
    )
    parser.add_argument(
        "--agentx_cancellation_timeout",
        type=float,
        default=10.0,
        help="Seconds to drain cancellation and transfer ownership.",
    )
    parser.add_argument(
        "--sticky_sessions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain bounded chat affinity with the ordinary DP placement policy.",
    )
    return parser
