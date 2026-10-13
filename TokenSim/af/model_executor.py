"""Whole-model execution on the same AF DAG, providers and physical resources."""

from dataclasses import replace

from TokenSim.af.executor import AFExecutor
from TokenSim.af.latency import AFEndpoint
from TokenSim.af.planning import AFExecutionDAG, ExecutionNode, RankLocalWork
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError


class ModelAFExecutor(AFExecutor):
    supports_sampled_token_readback = True

    def __init__(self, *args, model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model = model or getattr(self.compute_provider, "model", None)
        if model is None:
            raise ConfigurationError("whole-model AF execution requires a ModelSpec")
        if self.runtime_plan is None:
            raise ConfigurationError("whole-model AF execution requires a runtime plan")
        if (self.overhead_profile is not None and any(name in self.overhead_profile.document['costs']
                for name in ('engine_output_process_us','engine_runner_bookkeeping_us'))
                and (self.operation_profile is None or 'sampled_token_readback' not in self.operation_profile.document['operations'])):
            raise ConfigurationError('engine output processing requires sampled_token_readback completion')
        for domain in self.runtime_plan.domains:
            if (
                domain.capability == "af_attention"
                and model.vocab_size % domain.tensor_parallel_size
            ):
                raise ConfigurationError("vocab_size must divide the attention TP size")

    def submit(self, source_a_worker_id, requests, phase=None):
        requests = tuple(requests)
        actual_phase = phase or (
            "prefill" if requests and requests[0].is_prefill else "decode"
        )
        if self.model.attention_type == "mla" and actual_phase == "prefill":
            lengths = {request.prefill_len for request in requests}
            if len(lengths) != 1 or any(
                getattr(request, "prefill_compute_len", request.prefill_len)
                != request.prefill_len
                or request.context_len != request.prefill_len
                for request in requests
            ):
                raise ConfigurationError(
                    "MLA supports only full equal-length, unchunked prefill"
                )
        return super().submit(source_a_worker_id, requests, phase)

    def _boundary_nodes(self, block, operation):
        bindings = self.runtime_plan.bindings_for_instance(
            block.placement.a_domain, block.placement.a_instance
        )
        tp = len(bindings)
        tokens = (
            block.request_count
            if block.phase == "prefill" and operation in {"lm_head", "token_selection"}
            else block.token_count
        )
        shape = (
            ("tokens", tokens),
            ("hidden_size", self.hidden_size),
            ("global_hidden_size", self.hidden_size),
            ("tp_size", tp),
        )
        return tuple(
            ExecutionNode(
                node_id=f"i{block.iteration_id}:b{block.block_id}:boundary:{operation}:r{binding.domain_rank}",
                operation=operation,
                kind="compute",
                predecessor_ids=(),
                stage="layout",
                direction=None,
                source_domain=binding.domain,
                target_domain=binding.domain,
                logical_participants=(f"{binding.domain}:r{binding.domain_rank}",),
                domain_ranks=(binding.domain_rank,),
                physical_workers=(binding.physical_identity,),
                dp_rank=binding.dp_rank,
                tp_rank=binding.tp_rank,
                ep_rank=binding.ep_rank,
                tp_group=binding.tp_group,
                ep_group=binding.ep_group,
                local_work=RankLocalWork(token_count=tokens, tensor_shape=shape),
            )
            for binding in bindings
        )

    def _boundary_collective(self, block, name, operation, after_stage, payload):
        bindings = self.runtime_plan.bindings_for_instance(
            block.placement.a_domain, block.placement.a_instance
        )
        if len(bindings) <= 1:
            return None
        ranks = tuple(binding.domain_rank for binding in bindings)
        candidates = [
            plan
            for plan in self.runtime_plan.collective_plans
            if plan.domain == block.placement.a_domain
            and plan.operation == operation
            and plan.after_stage == after_stage
            and set(plan.participant_ranks) == set(ranks)
        ]
        if len(candidates) != 1:
            raise ConfigurationError(
                f"boundary {name} requires one {operation} plan for ranks={ranks}"
            )
        plan = candidates[0]
        endpoints = tuple(
            AFEndpoint(
                binding.domain,
                binding.domain_rank,
                binding.physical_identity,
                binding.hardware,
                binding.dp_rank,
                binding.tp_rank,
                binding.ep_rank,
            )
            for binding in bindings
        )
        return AFExecutionDAG(
            (
                ExecutionNode(
                    node_id=f"i{block.iteration_id}:b{block.block_id}:boundary:{name}",
                    operation=operation,
                    kind="communication",
                    predecessor_ids=(),
                    stage=None,
                    direction="collective",
                    source_domain=plan.domain,
                    target_domain=plan.domain,
                    logical_participants=tuple(
                        f"{plan.domain}:r{rank}" for rank in ranks
                    ),
                    domain_ranks=ranks,
                    physical_workers=tuple(b.physical_identity for b in bindings),
                    dp_rank=bindings[0].dp_rank,
                    network=plan.network,
                    resource_group=plan.resource_group,
                    resource_lane=plan.resource_lane,
                    payload_bytes=payload,
                    participant_endpoints=endpoints,
                    logical_input_bytes_per_rank=tuple(
                        (rank, payload) for rank in ranks
                    ),
                ),
            )
        )

    def _run_model_dag(self, block, layer, link, dag, previous):
        offset = len(self.trace.events) if self.trace.enabled else 0
        yield from self._run_dag(block, layer, link, dag)
        if self.trace.enabled and previous:
            # Only this DAG's newly emitted roots; unrelated blocks may interleave.
            for index in range(offset, len(self.trace.events)):
                event = self.trace.events[index]
                if (
                    event.iteration_id == block.iteration_id
                    and event.block_id == block.block_id
                    and event.event_type != "host_prepare"
                    and (not event.predecessor_ids or all(
                        p.startswith(f"i{block.iteration_id}:prepare:")
                        or p == f"i{block.iteration_id}:control:payload:transport"
                        or p == f"i{block.iteration_id}:control:payload:host_submit"
                        for p in event.predecessor_ids
                    ))
                ):
                    self.trace.events[index] = replace(event, predecessor_ids=tuple(
                        dict.fromkeys((*previous, *event.predecessor_ids))))
        return dag.terminal_node_ids

    def _run_block(self, block):
        link = self._links[block.link_name]
        previous = ()
        bytes_per_element = dtype_size_bytes(self.dtype, role="activation_dtype")
        bindings = self.runtime_plan.bindings_for_instance(
            block.placement.a_domain, block.placement.a_instance
        )
        previous = yield from self._run_model_dag(
            block,
            -1,
            link,
            AFExecutionDAG(self._boundary_nodes(block, "embedding_lookup")),
            previous,
        )
        dag = self._boundary_collective(
            block,
            "embedding_reduce",
            "all_reduce",
            "a_compute",
            block.token_count * self.hidden_size * bytes_per_element,
        )
        if dag is not None:
            previous = yield from self._run_model_dag(block, -1, link, dag, previous)
        for layer in range(self.total_layers):
            dag = self.plan_builder.build_layer(block, layer, link)
            previous = yield from self._run_model_dag(block, layer, link, dag, previous)
        for operation in ("final_norm", "lm_head"):
            previous = yield from self._run_model_dag(
                block,
                -1,
                link,
                AFExecutionDAG(self._boundary_nodes(block, operation)),
                previous,
            )
        tokens = block.request_count if block.phase == "prefill" else block.token_count
        dag = self._boundary_collective(
            block,
            "logits_gather",
            "all_gather",
            "residual",
            tokens * (self.model.vocab_size // len(bindings)) * bytes_per_element,
        )
        if dag is not None:
            previous = yield from self._run_model_dag(block, -1, link, dag, previous)
        previous = yield from self._run_model_dag(
            block,
            -1,
            link,
            AFExecutionDAG(self._boundary_nodes(block, "token_selection")),
            previous,
        )
        if self.operation_profile is not None and 'sampled_token_readback' in self.operation_profile.document['operations']:
            # The native supported path is 1A1F, single in-flight decode.
            # Sampling's device completion is a predecessor, never fitted into
            # the ready-input readback constant. Iteration completion follows it.
            identity = bindings[0].physical_identity
            output = yield from self._control_stage(
                block, self._boundary_nodes(block, 'token_selection')[0],
                'sampled_token_readback', previous, identity, 0, device=True,
                node_id=f'i{block.iteration_id}:output:sampled_token_readback')
            previous = (output,)
        bookkeeping_cost = self._host_cost_us('engine_runner_bookkeeping_us', block)
        if bookkeeping_cost:
            node_id=f'i{block.iteration_id}:output:bookkeeping'
            yield from self._run_communication_host_stage(
                block,-1,self._boundary_nodes(block,'token_selection')[0],node_id,
                previous,(bindings[0].physical_identity,),bookkeeping_cost/1e6,'host_prepare',
                details={'cost_field':'engine_runner_bookkeeping_us','cost_scope':'per_step_per_worker'})
            previous=(node_id,)
        output_cost = self._host_cost_us('engine_output_process_us', block)
        if output_cost:
            yield from self._run_communication_host_stage(
                block,-1,self._boundary_nodes(block,'token_selection')[0],
                f'i{block.iteration_id}:output:process',previous,(bindings[0].physical_identity,),
                output_cost/1e6,'host_prepare',details={'cost_field':'engine_output_process_us','cost_scope':'per_step_per_worker'})
        self.active_iterations[block.iteration_id].block_completed_at[
            block.block_id
        ] = self.env.now
