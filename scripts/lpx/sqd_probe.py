"""Capacity checks, necessary bounds and closed-loop global-core replay."""

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from scripts.lpx.mfu_metrics import (
    critical_path_seconds,
    resource_demands,
    useful_matrix_flops,
)
from scripts.lpx.mfu_replay import replay_cohorts
from scripts.lpx.replay_profiles import save_profile, source_identity
from scripts.lpx.sqd_memory import gpu_memory
from scripts.lpx.sqd_program import GlobalCoreProgram


def pool_bound(profiles, gpu_costs, *, resident, batch, replicas):
    """Keep common LP30 demand and each symmetric global-GPU pool's service.

    Replica completion rates may differ. No equal-throughput mix is assumed.
    GPU cross-uplink contention is relaxed; common LP30 demands must agree.
    """
    for name, value in (
        ("resident", resident),
        ("batch", batch),
        ("replicas", replicas),
    ):
        integer(value, f"SQD bound {name}", minimum=1)
    if len(profiles) != replicas or resident % batch:
        raise ConfigurationError(
            "SQD bound requires complete cohorts and replica profiles"
        )
    common = []
    for profile in profiles:
        demand = resource_demands(profile.nodes, include_lifetimes=True)
        common.append(
            {
                key: value
                for key, value in demand.items()
                if not key.startswith("sqd/gpu")
            }
        )
    if any(demand != common[0] for demand in common[1:]):
        raise ConfigurationError("SQD bound requires identical common LP30 demand")
    busiest = max(common[0], key=common[0].get)
    lp_limit = batch / common[0][busiest]
    gpu_limit = min(replicas * batch / row["seconds"] for row in gpu_costs["layers"])
    latency = min(critical_path_seconds(profile.nodes) for profile in profiles)
    return dict(
        aggregate_tps=min(lp_limit, gpu_limit, resident / latency),
        lp30_resource_limited_tps=lp_limit,
        gpu_pool_limited_tps=gpu_limit,
        latency_limited_tps=resident / latency,
        critical_path_s=latency,
        busiest_common_resource=busiest,
        common_resource_seconds_per_forward=common[0][busiest],
        scope="Necessary bound for this whole-buffer protocol with lifetime holds; GPU cross-uplink contention and physical compiler placement relaxed.",
    )


def probe(
    memory,
    *,
    pp,
    gpu_device="b200_sxm",
    gpu_tp=4,
    replicas=1,
    resident=None,
    request_limit=256,
    warmup=8,
    measured=32,
    bounds_only=False,
    profile_folder=None,
    spacing_us=0.0,
    pace_us=0.0,
    **program_settings,
):
    info = (
        memory.maximum(pp, request_limit)
        if resident is None
        else memory.select(pp, resident)
    )
    if info is None:
        raise ConfigurationError(
            "SQD LP30 placement cannot admit the resident requests"
        )
    resident = info["resident_requests"]
    gpu_ledger = gpu_memory(
        memory.hardware,
        memory.model,
        device=gpu_device,
        tp=gpu_tp,
        replicas=replicas,
        resident=resident,
        batch=memory.batch,
    )
    builder = GlobalCoreProgram(
        memory.hardware,
        memory.model,
        lp_tp=memory.tp,
        counts=info["layer_counts"],
        gpu_device=gpu_device,
        gpu_tp=gpu_tp,
        replicas=replicas,
        batch=memory.batch,
        **program_settings,
    )
    profiles = [builder.build(replica) for replica in range(replicas)]
    bounds = pool_bound(
        profiles,
        builder.gpu_cost,
        resident=resident,
        batch=memory.batch,
        replicas=replicas,
    )
    work = useful_matrix_flops(memory.model, 101000)
    peak = (
        builder.lp_cards
        * memory.hardware.device("groq_lp30").peak_compute_for("fp8").value
        + builder.gpu_cards
        * memory.hardware.device(gpu_device).peak_compute_for("fp8").value
    )
    bounds["mfu_percent"] = 100 * bounds["aggregate_tps"] * work["total"] / peak
    identity = dict(
        lp_tp=memory.tp,
        pp=pp,
        gpu_device=gpu_device,
        gpu_tp=gpu_tp,
        replicas=replicas,
        batch=memory.batch,
        resident=resident,
        program_settings=program_settings,
        source_sha256=source_identity(),
    )
    if profile_folder is not None:
        profiles = [
            save_profile(
                profile_folder / f"replica-{i}.json.gz", p, identity | dict(replica=i)
            )
            for i, p in enumerate(profiles)
        ]
    result = dict(
        status="bounded" if bounds_only else "ok",
        case=identity,
        lp30_cards=builder.lp_cards,
        gpu_cards=builder.gpu_cards,
        total_cards=builder.lp_cards + builder.gpu_cards,
        lp30_memory=info,
        gpu_memory=gpu_ledger,
        useful_flops_per_token=work,
        allocated_fp8_peak_flops_per_s=peak,
        profiles=[p.metadata for p in profiles],
        bounds=bounds,
        request_search_limit_reached=resident == request_limit,
        scope="Fixed 101K independent decode; separate FP8 KV with 102K capacity, native cost projection and whole-buffer operator scheduling; mixed-ownership compiler, physical banks and fabric unvalidated.",
    )
    if not bounds_only:
        replay = replay_cohorts(
            profiles[0],
            additional_profiles=profiles[1:],
            resident_requests=resident,
            batch=memory.batch,
            warmup_rounds=warmup,
            measured_rounds=measured,
            initial_spacing_s=spacing_us * 1e-6,
            issue_interval_s=pace_us * 1e-6,
        )
        result["replay"] = replay
        result["mfu_percent"] = 100 * replay["aggregate_tps"] * work["total"] / peak
    return result
