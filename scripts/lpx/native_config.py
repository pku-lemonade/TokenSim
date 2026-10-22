"""Map declared LPX layouts to validated production AF physical ranks."""

from TokenSim.af.factory import build_model_runtime
from TokenSim.config.validation import boolean
from TokenSim.errors import ConfigurationError


def session_prefix_context(
    workload, case, service, *, handoff_mode, warmup_mode, warming
):
    if warming and warmup_mode == "resident_decode":
        return workload.cached_tokens + workload.new_tokens
    if service == "target" and case.prefill and handoff_mode == "full":
        return 0
    return workload.cached_tokens


def runtime_configurations(
    hardware, target, case, *, draft_spec=None, a_dp=1, verification_width=1, **settings
):
    layouts = [("target", target, case.decode, a_dp)]
    if case.prefill:
        layouts.append(("prefill", target, case.prefill, 1))
    if case.draft:
        if draft_spec is None:
            raise ConfigurationError(
                "external drafting requires an explicit draft model"
            )
        layouts.append(("draft", draft_spec, case.draft, case.draft_a_replicas))
    return {
        name: configuration(
            hardware,
            model,
            layout,
            namespace=name,
            a_dp=replicas,
            **{
                **settings,
                "micro_batch": settings.get("micro_batch", 1)
                * (verification_width if name == "target" else 1),
            },
        )
        for name, model, layout, replicas in layouts
    }


def configuration(
    hardware,
    model,
    layout,
    *,
    namespace,
    micro_batch=1,
    a_dp=1,
    execution_mode="cuda_graph",
    graph_kernel_us=0.2,
    endpoint_transfer_occupancy="wire",
    dag_template_cache=False,
    colocated_dense_reuse=False,
    pipeline_layer_counts=None,
    ffn_batching=False,
    ffn_batch_wait_us=5.0,
):
    boolean(colocated_dense_reuse, "colocated dense reuse")
    policy = hardware.device(layout.a_device).analytical.get("lpx_pipeline")
    if policy and policy.get("activation_dtype", "bf16") != model.activation_dtype:
        raise ConfigurationError("LP30 pipeline activation dtype must match the model")
    if layout.a_cp != 1:
        raise ConfigurationError(
            "context parallelism needs a native distributed-attention plan"
        )
    if layout.colocated and layout.f_tp != 1 and (layout.f_ep != 1 or a_dp != 1):
        raise ConfigurationError(
            "colocated expert TP with EP or multiple attention replicas needs "
            "a complete-model prefill backend supporting hybrid expert TP/EP"
        )
    a_tp, a_pp = layout.a_tp, layout.a_pp
    f_tp, f_pp, ep = layout.f_tp, layout.f_pp, layout.f_ep
    f_dp = ep * a_dp if layout.colocated else ep
    a_count, f_count = a_tp * a_pp * a_dp, f_tp * f_pp * f_dp
    if layout.colocated and (a_count != f_count or a_pp != f_pp):
        raise ConfigurationError(
            "colocated native domains require matching physical inventories"
        )
    a_name, f_name = f"{namespace}-a", (
        f"{namespace}-a" if layout.colocated else f"{namespace}-f"
    )
    nets = {
        f"{namespace}-af": (
            hardware.device(layout.a_device).scale_up_link
            if layout.colocated
            else "ethernet1600Gb"
        ),
        f"{namespace}-a-up": hardware.device(layout.a_device).scale_up_link,
        f"{namespace}-f-up": hardware.device(layout.f_device).scale_up_link,
    }
    if layout.a_device == layout.f_device and not layout.colocated:
        nets[f"{namespace}-af"] = hardware.device(layout.a_device).scale_up_link
    groups = [
        dict(
            name=a_name,
            capabilities=(
                ["af_attention", "af_ffn"] if layout.colocated else ["af_attention"]
            ),
            hardware=layout.a_device,
            num_workers=a_count,
            network=f"{namespace}-af",
        )
    ]
    if not layout.colocated:
        groups.append(
            dict(
                name=f_name,
                capabilities=["af_ffn"],
                hardware=layout.f_device,
                num_workers=f_count,
                network=f"{namespace}-af",
            )
        )
    domains = []
    for name, group, capability, tp, pp, dp, expert_size in (
        ("a", a_name, "af_attention", a_tp, a_pp, a_dp, 1),
        ("f", f_name, "af_ffn", f_tp, f_pp, f_dp, ep),
    ):
        # F reuses A's base coordinates while retaining its declared expert
        # tensor partition and spreading its EP groups over A replicas.
        if name == "f" and layout.colocated:
            tp, dp, expert_size = a_tp, a_dp, ep * a_dp
        instances = []
        for replica in range(dp):
            mappings = []
            for stage in range(pp):
                for rank in range(tp):
                    index = replica * pp * tp + stage * tp + rank
                    expert_tp = tp * dp // expert_size if name == "f" else tp
                    mappings.append(
                        dict(
                            domain_rank=index,
                            tp_rank=rank,
                            pp_rank=stage,
                            dp_rank=replica,
                            ep_rank=(
                                (replica * tp + rank) // expert_tp
                                if name == "f" and expert_size > 1
                                else 0
                            ),
                            physical_worker=dict(group_name=group, worker_index=index),
                        )
                    )
            instances.append(dict(name=f"{name}{replica}", rank_mapping=mappings))
        plans = []
        for operation, after in (
            (("all_reduce", "a_compute"), ("all_gather", "residual"))
            if name == "a"
            else (("all_reduce", "expert_compute"),)
        ):
            plans.append(
                dict(
                    operation=operation,
                    after_stage=after,
                    participants=list(range(tp)),
                    network=f"{namespace}-{name}-up",
                    resource_group=f"{name}-up",
                    resource_lane="collective",
                    algorithm=getattr(layout, f"{name}_collective_algorithm"),
                )
            )
        domains.append(
            dict(
                name=f"{namespace}-{name}-domain",
                capability=capability,
                worker_group=group,
                parallel=dict(
                    tensor_parallel_size=tp,
                    pipeline_parallel_size=pp,
                    data_parallel_size=dp,
                    expert_parallel_size=expert_size,
                    **(
                        {"pipeline_layer_counts": pipeline_layer_counts}
                        if pipeline_layer_counts is not None
                        else {}
                    ),
                ),
                logical_instances=instances,
                communication_plan=plans,
            )
        )
    fabrics = [
        dict(
            resource_group="af",
            network=f"{namespace}-af",
            duplex="full",
            lanes=[dict(name="a_to_f", capacity=1), dict(name="f_to_a", capacity=1)],
        )
    ]
    for name in ("a", "f"):
        fabrics.append(
            dict(
                resource_group=f"{name}-up",
                network=f"{namespace}-{name}-up",
                duplex="half",
                lanes=[dict(name="collective", capacity=1)],
            )
        )
    doc = dict(
        num_workers=sum(group["num_workers"] for group in groups),
        topology="groq_lpx256" if layout.a_device == "groq_lp30" else "rubin_nvl72",
        networks=nets,
        worker_groups=groups,
        af=dict(
            enabled=True,
            af_block_size=micro_batch,
            transfer_topology_mode="flat_link",
            communication_resource_scope="endpoint",
            endpoint_transfer_occupancy=endpoint_transfer_occupancy,
            communication_domains=dict(
                node_size=8,
                node_count=sum((group["num_workers"] + 7) // 8 for group in groups),
                placement_strategy="dedicated",
            ),
            router_location="a",
            latency_provider="operator_table",
            trace_enabled=True,
            compute_execution_mode=execution_mode,
            graph_device_kernel_us=graph_kernel_us,
            coalesce_ready_transfers=True,
            ffn_batching=ffn_batching,
            ffn_batch_wait_us=ffn_batch_wait_us,
            dag_template_cache_max_tokens=32,
            dag_template_cache=dag_template_cache,
            activation_dtype=model.activation_dtype,
            kv_cache_dtype=model.kv_cache_dtype,
            execution_domains=domains,
            communication_fabrics=fabrics,
            links=[
                dict(
                    name=f"{namespace}-link",
                    attention_group=a_name,
                    ffn_group=f_name,
                    attention_domain=domains[0]["name"],
                    ffn_domain=domains[1]["name"],
                    network=f"{namespace}-af",
                    resource_group="af",
                    duplex="full",
                    a_to_f_capacity=1,
                    f_to_a_capacity=1,
                    colocated=layout.colocated,
                    colocated_dense_reuse=colocated_dense_reuse and layout.colocated,
                )
            ],
        ),
    )
    return build_model_runtime(doc, hardware=hardware, model=model)
