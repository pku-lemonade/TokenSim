"""Eager DP=1 control protocol, separate from activation and routing data."""
import json
from dataclasses import replace

from TokenSim.af.latency import AFServiceEstimate
from TokenSim.af.planning import ExecutionNode
from TokenSim.af.trace import AFTraceEvent


def encode_eager_control(token_count):
    """Match the reviewed AFD compact, sorted JSON schema for one eager stage."""
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("native control token count must be a positive integer")
    payload = {
        "dp_metadata_list": {"0": {
            "num_tokens_across_dp_cpu": [token_count],
            "max_tokens_across_dp_cpu": token_count,
        }},
        "is_graph_capturing": False,
        "is_warmup": False,
        "is_graph_replaying": False,
        "is_profile": False,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


class NativeControlMixin:
    def _control_nodes(self, block):
        link = self._links[block.link_name]
        a = self.runtime_plan.bindings_for_instance(
            block.placement.a_domain, block.placement.a_instance)[0]
        f = self.runtime_plan.bindings_for_instance(
            block.placement.f_domain, block.placement.f_instance)[0]
        prefix = f"i{block.iteration_id}:control"
        payload = encode_eager_control(block.token_count)
        nodes = []
        for name, size, dtype in (("length", 8, "int64"), ("payload", len(payload), "uint8")):
            nodes.append(ExecutionNode(
                node_id=f"{prefix}:{name}", operation=f"af_control_{name}",
                kind="communication", predecessor_ids=(), stage=None, direction="a_to_f",
                source_domain=a.domain, target_domain=f.domain,
                logical_participants=(f"{a.domain}:r{a.domain_rank}", f"{f.domain}:r{f.domain_rank}"),
                domain_ranks=(a.domain_rank, f.domain_rank),
                physical_workers=(a.physical_identity, f.physical_identity), dp_rank=0,
                network=link.network, resource_group=link.resource_group,
                payload_bytes=size, metadata={"message_class": "control", "wire_dtype": dtype,
                    "transport_backend": "torch_distributed_nccl", "control_schema": "afd-eager-dp1-v1",
                    "payload_utf8": payload.decode() if name == "payload" else None}))
        return tuple(nodes)

    def _control_stage(self, block, template, name, predecessors, identity, duration, *, device=False, node_id=None):
        node_id = node_id or f"i{block.iteration_id}:control:{name}"
        stage = replace(template, operation=name)
        profile = self.operation_profile
        if device and profile is not None and name in profile.document['operations']:
            entry = profile.document['operations'][name]
            keyed = self.resources.host_resources_for((identity,))
            estimate = AFServiceEstimate(entry['fixed_us']/1e6, f'operation-profile:{profile.fingerprint}:{name}',
                details={'timing_scope': 'operation_service', 'coverage': entry['coverage'],
                         'resource_model': entry['resource_model'], 'input_ready': entry['input_ready'],
                         'completion': entry['completion'], 'effective_cost_us': entry['fixed_us'],
                         'calibrated': profile.document['evidence']['kind'] == 'measured',
                         'profile_id': profile.document['profile_id'],
                         'limitation': 'calling thread approximation; independent copy engine contention unmodeled'})
            ready, start, end, _ = yield from self._run_resources_stage(
                block, -1, node_id, tuple(r for _, r in keyed), lambda: estimate)
            if self.trace.enabled:
                self.trace.record(AFTraceEvent(
                    iteration_id=block.iteration_id, block_id=block.block_id, layer_id=-1,
                    phase=block.phase, event_type='operation_service', operation=name,
                    resource_id=keyed[0][0], resource_ids=tuple(k for k, _ in keyed),
                    ready_at=ready, start_at=start, end_at=end, queue_duration=start-ready,
                    service_duration=end-start, request_count=block.request_count,
                    token_count=block.token_count, source_id=estimate.source_id,
                    details={**estimate.details, 'message_class': ('engine_output' if name == 'sampled_token_readback' else
                        'engine_prepare' if name.startswith('a_') else 'control')}, node_id=node_id,
                    predecessor_ids=predecessors, physical_endpoints=(identity,)))
            self.active_stages.pop((block.iteration_id, block.block_id, node_id), None)
        elif not device:
            yield from self._run_communication_host_stage(
                block, -1, stage, node_id, predecessors, (identity,), duration / 1e6,
                "host_prepare", details={"message_class": "control", "cost_scope": name,
                                        "cost_field": f"control_{name}_us",
                                        "calibrated": False})
        else:
            keyed = self.resources.compute_resources_for((identity,))
            estimate = AFServiceEstimate(duration / 1e6, "assumption:control-device-service",
                details={"timing_scope": "device_service", "cost_scope": name, "calibrated": False})
            ready, start, end, _ = yield from self._run_resources_stage(
                block, -1, node_id, tuple(r for _, r in keyed), lambda: estimate)
            if self.trace.enabled:
                self.trace.record(AFTraceEvent(
                    iteration_id=block.iteration_id, block_id=block.block_id, layer_id=-1,
                    phase=block.phase, event_type="control_device", operation=name,
                    resource_id=f"worker:{keyed[0][0]}",
                    resource_ids=tuple(f"worker:{key}" for key, _ in keyed),
                    ready_at=ready, start_at=start, end_at=end, queue_duration=start-ready,
                    service_duration=end-start, request_count=block.request_count,
                    token_count=block.token_count, source_id=estimate.source_id,
                    details={**estimate.details, "message_class": "control"},
                    node_id=node_id, predecessor_ids=predecessors,
                    physical_endpoints=(identity,)))
            self.active_stages.pop((block.iteration_id, block.block_id, node_id), None)
        return node_id

    def _run_native_control(self, block, state):
        if self.cluster_config.af.communication_timing.native_call_order_enabled:
            yield from self._run_native_order_control(block, state)
            return
        timing = self.cluster_config.af.communication_timing
        length, payload = self._control_nodes(block)
        a, f = length.physical_workers
        prepared = yield from self._wait_step_preparation(block, (a,))
        encoded = yield from self._control_stage(block, length, "encode", prepared,
            a, self._host_cost_us("control_encode_us", block))
        staged = yield from self._control_stage(block, length, "h2d", (encoded,),
            a, timing.control_h2d_us, device=True)
        link = self._links[block.link_name]
        yield from self._run_planned_communication(block, -1, link,
            replace(length, predecessor_ids=(staged,)))
        read_size = yield from self._control_stage(block, length, "length_readback",
            (length.node_id,), f, timing.control_length_readback_us, device=True)
        # The receiving allocation depends on the CPU-visible length. The
        # aggregate delivery model starts payload transport only after this.
        yield from self._run_planned_communication(block, -1, link,
            replace(payload, predecessor_ids=(read_size,)), transport_completed=state["a_ready"])
        copied = yield from self._control_stage(block, payload, "payload_readback",
            (payload.node_id,), f, timing.control_payload_readback_us, device=True)
        decoded = yield from self._control_stage(block, payload, "decode", (copied,),
            f, self._host_cost_us("control_decode_us", block))
        if self.operation_profile is not None and 'f_dp_metadata' in self.operation_profile.document['operations']:
            state['f_node'] = yield from self._control_stage(block,payload,'f_dp_metadata',(decoded,),f,0,device=True)
        state["f_ready"].succeed()
