"""Native Gemma decode probes with admission, scheduling and MFU evidence."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.mfu_capacity import ResidentCapacity
from scripts.lpx.mfu_metrics import throughput_bounds, useful_matrix_flops
from scripts.lpx.mfu_replay import replay_cohorts
from scripts.lpx.native_config import configuration
from scripts.lpx.native_search import memory_report, streaming_settings
from scripts.lpx.partition import native_footprint, partition_solutions
from scripts.lpx.replay_profiles import cached_profile, source_identity
from scripts.run_lpx_context_parallel import environment, layout_for


@dataclass(frozen=True)
class MFUCase:
    tp: int
    kvp: int
    pp: int
    batch: int = 1
    resident_requests: int | None = None
    request_search_limit: int = 256
    kv_parallel_layers: str = "full_attention"
    layer_counts: tuple | None = None
    initial_spacing_us: float = 0.0
    issue_interval_us: float = 0.0

    def __post_init__(self):
        for name in ("tp", "kvp", "pp", "batch", "request_search_limit"):
            integer(getattr(self, name), f"MFU case {name}", minimum=1)
        if self.resident_requests is not None:
            integer(self.resident_requests, "MFU case resident requests", minimum=1)
        if self.tp % self.kvp or self.kv_parallel_layers not in {
            "all",
            "full_attention",
        }:
            raise ConfigurationError("invalid MFU KV parallel layout")
        for name in ("initial_spacing_us", "issue_interval_us"):
            number(getattr(self, name), f"MFU case {name}")
        if self.layer_counts is not None and self.resident_requests is None:
            raise ConfigurationError("custom MFU partitions need a resident count")


def prepare(config, case):
    config = deepcopy(config)
    if "memory" in config:
        config["memory"]["prefill_chunk"] = max(
            config["memory"].get("prefill_chunk", 16), case.batch
        )
    hardware, model, _, _ = environment(config)
    return config, hardware, model


def runtime_for(hardware, model, case, pp, counts=None):
    return configuration(
        hardware,
        model,
        layout_for(case.kvp, pp, case.tp),
        namespace="mfu-search",
        micro_batch=case.batch,
        dag_template_cache=True,
        colocated_dense_reuse=True,
        pipeline_layer_counts=counts,
        kv_parallel_layers=case.kv_parallel_layers,
        kv_append_block=16,
    )


def select_partition(hardware, model, case, maximum_context):
    if case.layer_counts is not None:
        return case.resident_requests, case.layer_counts, None
    seed = runtime_for(hardware, model, case, model.num_layers)
    footprint = native_footprint(seed, maximum_context)
    resident = case.resident_requests
    screening = None
    if resident is None:
        screening = ResidentCapacity(footprint).frontier(
            case.request_search_limit, case.pp
        )
        maximum = screening["maximum_resident_by_pp"].get(case.pp, 0)
        resident = maximum // case.batch * case.batch
    if not resident or resident % case.batch:
        raise ConfigurationError("no complete MFU cohort fits the requested layout")
    scaled = replace(
        footprint,
        kv=tuple(tuple(value * resident for value in row) for row in footprint.kv),
    )
    solution = partition_solutions(scaled, case.pp).get(case.pp)
    if solution is None:
        raise ConfigurationError("no contiguous MFU partition fits resident KV")
    return resident, solution["layer_counts"], screening


def probe(
    config, case, cache, *, warmup_rounds=8, measured_rounds=32, bounds_only=False
):
    config, hardware, model = prepare(config, case)
    if case.pp > model.num_layers:
        raise ConfigurationError("MFU pipeline cannot exceed the model layer count")
    context, maximum = 101000, 102000
    resident, counts, screening = select_partition(hardware, model, case, maximum)
    runtime = runtime_for(hardware, model, case, case.pp, counts)
    memory = memory_report(runtime, maximum, resident)
    layout = layout_for(case.kvp, case.pp, case.tp)
    profile = cached_profile(
        runtime,
        "mfu-search",
        cache,
        batch=case.batch,
        queries=1,
        phase="decode",
        emit_logits=True,
        context=context,
        contexts=(context,) * case.batch,
        maximum_contexts=(maximum,) * case.batch,
        dp_rank=0,
        seed=42,
        sample=0,
        streaming=streaming_settings(layout, launch_us=0),
    )
    limits = throughput_bounds(
        profile.nodes, resident_requests=resident, batch=case.batch
    )
    work = useful_matrix_flops(model, context)
    chips = case.tp * case.pp
    device = hardware.device("groq_lp30")
    peak = device.peak_compute_for("fp8").value * chips
    bf16_peak = device.peak_compute_for("bf16").value * chips
    result = dict(
        status="bounded" if bounds_only else "ok",
        case=asdict(case),
        config=config,
        source_sha256=source_identity(),
        chips=chips,
        resident_requests=resident,
        layer_counts=counts,
        memory=memory,
        screening=screening,
        profile=profile.metadata,
        refined_nodes=len(profile.nodes),
        isolated_step_s=profile.metadata["modeled_duration_s"],
        useful_flops_per_token=work,
        rack_fp8_peak_flops_per_s=peak,
        bounds={
            **limits,
            "mfu_percent": 100 * limits["aggregate_tps"] * work["total"] / peak,
        },
        multirack_network_unvalidated=chips > 256,
        simulation_scope="Fixed 101K context, FP8 weights and KV, BF16 attention, resident 102K KV, no speculation or external-memory offload.",
    )
    if bounds_only:
        return result
    replay = replay_cohorts(
        profile,
        resident_requests=resident,
        batch=case.batch,
        warmup_rounds=warmup_rounds,
        measured_rounds=measured_rounds,
        initial_spacing_s=case.initial_spacing_us * 1e-6,
        issue_interval_s=case.issue_interval_us * 1e-6,
    )
    tps = replay["aggregate_tps"]
    excess = (
        replay["measured_tokens"] - limits["aggregate_tps"] * replay["measured_seconds"]
    )
    if excess > resident + 1e-6:
        raise SimulationStateError(
            "finite replay exceeds its steady bound and in-flight token allowance"
        )
    result.update(
        replay=replay,
        mfu_percent=100 * tps * work["total"] / peak,
        mfu_with_bf16_peak_correction_percent=100
        * tps
        * (work["linear"] / peak + work["attention"] / bf16_peak),
    )
    return result
