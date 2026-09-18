"""Full V3 BF16 decode simulation, including model input/output operations.

CPU only. No checkpoint, prefill, MTP, or serving engine is executed.
Default virtual topology: 1A32F; F leader Router; A-side Combine.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, fields
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import simpy
from TokenSim.af import AFExecutor, AFServiceEstimate
from TokenSim.af.trace import AFTraceRecorder
from TokenSim.af.planning import ExecutionNode
from TokenSim.moe.routing import ExpertRouting, RoutingAssignment
from TokenSim.operator_data import (
    Calibration, EvidenceGrade, HardwareProfile, ModelOperatorManifest,
    Observation, OperatorDataComputeLatencyProvider, OperatorDataGenerator,
    OperatorDataLookup, OperatorDataPackage, OperatorDataTransferLatencyProvider,
    SourceRecord,
)
from TokenSim.operator_data.schema import TABLE_SPECS
from scripts.run_deepseek_mla import configuration, legacy_coverage_shapes, write_json


def signature(table, key):
    return table, json.dumps(key, sort_keys=True)


class RecordingLookup:
    """Discover exact shapes or audit actual measured/theoretical lookups."""
    def __init__(self, hardware, communication, lookup=None):
        self.hardware = hardware
        self.communication = communication
        self.lookup_impl = lookup
        self.package = (lookup.package if lookup else
                        SimpleNamespace(dataset_version="discovery-not-results"))
        self.shapes = {}
        self.calls = Counter()
        self.methods = Counter()

    def lookup(self, table, key):
        from TokenSim.operator_data import LookupResult
        # The explicit diagnostic alias changes F capacity only. All latency
        # lookups retain the measured A800 compute/bandwidth profile.
        if key.get('device') == self.hardware['device'] + '-hypothetical-f-memory':
            key = dict(key, device=self.hardware['device'])
        sig = signature(table, key)
        self.shapes[sig] = dict(table=table, key=dict(key))
        self.calls[sig] += 1
        if self.lookup_impl is None:
            return LookupResult(1.0, "exact", "discovery", key)
        result = self.lookup_impl.lookup(table, key)
        self.methods[(table, result.values.get("timing_method", "unspecified"))] += 1
        return result

    def link_profile(self, key):
        from TokenSim.operator_data import LookupResult
        if self.lookup_impl is not None:
            return self.lookup("communication", key)
        self.lookup("communication", key)
        # All virtual intra-F traffic explicitly shares the F-to-A fabric lane.
        lane = "a_to_f" if key["resource_lane"] == "a_to_f" else "f_to_a"
        return LookupResult(0.0, "exact", "discovery", key,
            values=dict(self.communication["transport"][lane], timing_scope="transport_service"))


def boundary_shapes(device, tokens, hidden, vocab, a_tp=1):
    if a_tp < 1 or vocab % a_tp:
        raise ValueError('Vocabulary must divide A TP evenly')
    common = dict(device=device, backend="analytical", activation_dtype="bf16")
    def elementwise(op, width):
        return dict(common, op_name=op, source_dtype="bf16", target_dtype="bf16",
                    num_tokens=tokens, hidden_size=width)
    return (
        ("embedding", "elementwise", elementwise("embedding_lookup", hidden)),
        ("final_norm", "elementwise", elementwise("rmsnorm", hidden)),
        ("lm_head", "gemm", dict(common, weight_dtype="bf16", m=tokens, n=vocab//a_tp, k=hidden)),
        ("greedy_argmax", "elementwise", elementwise("argmax", vocab)),
    )


class UniformRequestRouting(ExpertRouting):
    """Uniform distinct experts; invariant to AF microbatch boundaries/order."""
    def __init__(self, moe_config, seed=42):
        super().__init__(moe_config, seed=seed)
        self.seed = seed

    def _synthetic_assignments(self, request_tokens, request_owners, *,
                              iteration_id, phase, decode_step, layer_id):
        assignments = []
        top_k = self.moe_config.num_experts_per_tok
        for request, tokens in request_tokens.items():
            for token in tokens:
                identity = f'{self.seed}:{phase}:{decode_step}:{layer_id}:{request}:{token}'
                seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big')
                experts = random.Random(seed).sample(range(self.moe_config.num_experts), top_k)
                for k, expert in enumerate(experts):
                    assignments.append(RoutingAssignment(
                        iteration_id=iteration_id, phase=phase, decode_step=decode_step,
                        layer_id=layer_id, request_id=request, token_id=token,
                        top_k_index=k, expert_id=expert, routing_weight=1/top_k,
                        a_dp_rank=request_owners[request][0], a_tp_rank=request_owners[request][1]))
        return tuple(assignments)


class FullModelExecutor(AFExecutor):
    """Boundary operators contend on the same A GPU as transformer layers."""
    def __init__(self, *args, boundary_lookup, vocab_size, **kwargs):
        super().__init__(*args, **kwargs)
        if any(vocab_size % len(r.rank_keys) for r in self.runtime_plan.replicas if r.is_attention):
            raise ValueError("Vocabulary must divide A TP evenly")
        self.boundary_lookup = boundary_lookup
        self.vocab_size = vocab_size
        self.boundary_events = []
        self.expert_routing = UniformRequestRouting(self.expert_routing.moe_config)

    def _boundary(self, block, item, binding=None):
        name, table, key = item
        if binding is None:
            binding = self.runtime_plan.bindings_for_instance(
                block.placement.a_domain, block.placement.a_instance)[0]
        resources = self.resources.compute_resources_for((binding.physical_identity,))
        worker_id = resources[0][0]
        result = self.boundary_lookup.lookup(table, key)
        estimate = AFServiceEstimate(result.latency_us / 1e6, result.source_id,
            dataset_version=self.boundary_lookup.package.dataset_version)
        ready, start, end, _ = yield from self._run_resources_stage(
            block, -1 if name == "embedding" else self.total_layers,
            f"i{block.iteration_id}:b{block.block_id}:{name}:r{binding.domain_rank}",
            tuple(r for _, r in resources), lambda: estimate)
        if self.trace.enabled:
            self.boundary_events.append(dict(name=name, iteration_id=block.iteration_id,
                block_id=block.block_id, resource_id=f"worker:{worker_id}",
                tp_rank=binding.tp_rank,
                ready_at=ready, start_at=start, end_at=end,
                service_duration=end-start, queue_duration=start-ready,
                table=table, key=key, source_id=result.source_id,
                timing_method=result.values.get("timing_method", "discovery")))

    def _parallel_boundary(self, block, item, bindings):
        if len(bindings) == 1:
            yield from self._boundary(block, item, bindings[0])
        else:
            yield self.env.all_of([self.env.process(self._boundary(block, item, binding))
                                   for binding in bindings])

    def _boundary_collective(self, block, bindings, *, name, operation, input_bytes):
        if len(bindings) == 1:
            return
        link = self._links[block.link_name]
        node_id = f'i{block.iteration_id}:b{block.block_id}:{name}'
        ranks = tuple(b.domain_rank for b in bindings)
        output_bytes = input_bytes * (len(bindings) if operation == 'all_gather' else 1)
        step = ExecutionNode(node_id=node_id, operation=operation, kind='communication',
            predecessor_ids=(), stage=None, direction='tp_collective',
            source_domain=bindings[0].domain, target_domain=bindings[0].domain,
            logical_participants=(bindings[0].instance,), domain_ranks=ranks,
            physical_workers=tuple(b.physical_identity for b in bindings),
            dp_rank=bindings[0].dp_rank, tp_group=ranks, network=link.network,
            resource_group=link.resource_group,
            resource_lane='shared' if link.duplex == 'half' else 'a_to_f',
            payload_bytes=(math.ceil(2*(len(bindings)-1)/len(bindings)*input_bytes)
                           if operation == 'all_reduce' else (len(bindings)-1)*input_bytes),
            logical_input_bytes_per_rank=tuple((r,input_bytes) for r in ranks),
            logical_output_bytes_per_rank=tuple((r,output_bytes) for r in ranks),
            logical_stage_id=node_id,
            participant_endpoints=tuple(self.plan_builder._endpoint(b) for b in bindings),
            source_layout='partial' if operation == 'all_reduce' else 'sharded',
            target_layout='replicated')
        layer = -1 if name == 'embedding_all_reduce' else -2
        yield from self._run_planned_communication(block, layer, link, step)

    def _run_block(self, block):
        device = self.resources.worker(block.source_a_worker_id).hardware
        bindings = self.runtime_plan.bindings_for_instance(
            block.placement.a_domain, block.placement.a_instance)
        items = boundary_shapes(device, block.token_count, self.hidden_size, self.vocab_size, len(bindings))
        yield from self._parallel_boundary(block, items[0], bindings)
        yield from self._boundary_collective(block, bindings, name='embedding_all_reduce',
            operation='all_reduce', input_bytes=block.token_count*self.hidden_size*2)
        link = self._links[block.link_name]
        for layer in range(self.total_layers):
            dag = self.plan_builder.build_layer(block, layer, link)
            yield from self._run_dag(block, layer, link, dag)
        for item in items[1:3]:
            yield from self._parallel_boundary(block, item, bindings)
        yield from self._boundary_collective(block, bindings, name='logits_all_gather',
            operation='all_gather', input_bytes=block.token_count*(self.vocab_size//len(bindings))*2)
        yield from self._parallel_boundary(block, items[3], bindings)
        self.active_iterations[block.iteration_id].block_completed_at[block.block_id] = self.env.now


def simulate_full(config, lookup, batch, context, steps=1, trace=True):
    _, cluster, shape, moe, plan = config
    blocks = batch * ((context + steps - 1 + plan.block_size - 1) // plan.block_size)
    if any(blocks > r.logical_gpu_blocks for r in plan.replicas if r.is_attention):
        raise ValueError("Full 61-layer MLA cache exceeds virtual A capacity")
    env = simpy.Environment()
    executor = FullModelExecutor(env, cluster, cluster.workers(),
        OperatorDataComputeLatencyProvider(lookup, ModelOperatorManifest({shape.model_id: shape}), "analytical"),
        OperatorDataTransferLatencyProvider(lookup, "analytical"),
        shape.model_id, "bf16", plan.total_layers, shape.hidden_size,
        moe_config=moe, random_seed=42, runtime_plan=plan,
        trace_recorder=AFTraceRecorder(trace), boundary_lookup=lookup, vocab_size=129280)
    executor.trace.resource_capacities = executor.resources.resource_capacities()
    iterations = []
    for step in range(steps):
        requests = [SimpleNamespace(id=i, is_prefill=False, prefill_len=context,
            context_len=context+step, generation_idx=step) for i in range(batch)]
        iteration = executor.submit(0, requests, "decode")
        env.process(executor.run_iteration(iteration))
        env.run()
        iterations.append(iteration)
    if trace:
        validate_full_trace(executor, iterations)
    return executor, iterations


def validate_full_trace(executor, iterations):
    """Check boundary ordering and capacity together with the validated layer trace."""
    intervals = defaultdict(list)
    by_block = defaultdict(list)
    for e in executor.trace.events:
        by_block[(e.iteration_id, e.block_id)].append(e)
        if e.event_type == "compute" and e.service_duration > 0:
            intervals[e.resource_id].append((e.start_at, e.end_at))
    for e in executor.boundary_events:
        if e['service_duration'] > 0:
            intervals[e['resource_id']].append((e['start_at'], e['end_at']))
    for resource, spans in intervals.items():
        spans.sort()
        if any(a[1] > b[0] + 1e-10 for a, b in zip(spans, spans[1:])):
            raise ValueError(f"Full model compute overlap on {resource}")
    for iteration in iterations:
        for block in iteration.blocks:
            events = by_block[(iteration.iteration_id, block.block_id)]
            transformer = [e for e in events if e.layer_id >= 0]
            layers = {e.layer_id for e in transformer}
            if layers != set(range(executor.total_layers)):
                raise ValueError("Full model trace is missing transformer layers")
            boundary = defaultdict(list)
            for e in executor.boundary_events:
                if e['iteration_id'] == iteration.iteration_id and e['block_id'] == block.block_id:
                    boundary[e['name']].append(e)
            names = ['embedding', 'final_norm', 'lm_head', 'greedy_argmax']
            bindings = executor.runtime_plan.bindings_for_instance(
                block.placement.a_domain, block.placement.a_instance)
            if set(boundary) != set(names) or any(len(boundary[n]) != len(bindings) for n in names):
                raise ValueError("Full model trace is missing input/output operators")
            if any({e['tp_rank'] for e in boundary[n]} != {b.tp_rank for b in bindings} for n in names):
                raise ValueError("Model boundary does not cover every A rank exactly once")
            first = lambda name: min(e['start_at'] for e in boundary[name])
            last = lambda name: max(e['end_at'] for e in boundary[name])
            if last('embedding') > min(e.start_at for e in transformer) + 1e-10:
                raise ValueError("Transformer ran before embedding")
            if max(e.end_at for e in transformer) > first('final_norm') + 1e-10:
                raise ValueError("Output operators ran before transformer completed")
            for a, b in zip(names[1:], names[2:]):
                if last(a) > first(b) + 1e-10:
                    raise ValueError("Output operator dependency violated")
            if len(bindings) > 1:
                for layer, before, after in ((-1, last('embedding'), min(e.start_at for e in transformer)),
                                              (-2, last('lm_head'), first('greedy_argmax'))):
                    comm = [e for e in events if e.layer_id == layer]
                    if sum(e.event_type == 'collective' for e in comm) != 1:
                        raise ValueError('Missing model boundary collective')
                    if min(e.start_at for e in comm) < before - 1e-10 or max(e.end_at for e in comm) > after + 1e-10:
                        raise ValueError('Model boundary collective dependency violated')
            if abs(iteration.block_completed_at[block.block_id] - last('greedy_argmax')) > 1e-10:
                raise ValueError("Token marked complete before token selection")


def load_measurements(root, device):
    observations, sources = [], []
    for role in ('A', 'F'):
        path = root / f'compute-{role}' / 'measurements.json'
        contract_path = path.with_name('measurement-contract.json')
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        rows = json.loads(path.read_text(encoding='utf-8'))
        cases = path.with_name('cases.json')
        if contract['failed'] != 0 or contract['completed'] != len(rows):
            raise ValueError(f"Incomplete measurement contract: {role}")
        if hashlib.sha256(cases.read_bytes()).hexdigest() != contract['input_sha256']:
            raise ValueError(f"Measurement shape hash differs: {role}")
        source_id = f'measured:a800:{role}'
        sources.append(SourceRecord(source_id, EvidenceGrade.A, 'measured', device, 'analytical',
            str(contract_path), hashlib.sha256(path.read_bytes()).hexdigest()))
        for r in rows:
            if r['status'] != 'ok' or not r['validation']['passed'] or r['key']['device'] != device:
                raise ValueError(f"Invalid measurement: {role}")
            if r['split'] == 'runtime':
                observations.append(Observation(r['table'], r['key'], r['device_interval_us'], source_id))
    return observations, sources


def make_package(output, shapes, hardware, communication, observations, sources):
    hw = HardwareProfile(**{f.name: hardware[f.name] for f in fields(HardwareProfile) if f.name in hardware})
    source = SourceRecord('assumption:virtual-fabric', EvidenceGrade.D, 'assumption',
        hardware['device'], 'analytical', 'Configured shared fabric, including intra-F traffic', 'unmeasured')
    links = []
    for s in shapes:
        if s['table'] != 'communication':
            continue
        lane = 'a_to_f' if s['key']['resource_lane'] == 'a_to_f' else 'f_to_a'
        links.append(Observation('communication', dict(s['key'],
            **communication['transport'][lane], timing_scope='transport_service'), 0., source.source_id))
    calibration = Calibration(offset_us=hardware['launch_overhead_us'], calibration_id='ASSUMED-launch')
    generator = OperatorDataGenerator('deepseek-v3-full-decode-mixed-v1', {hw.device: hw},
        [source, *sources], calibrations={t: calibration for t in TABLE_SPECS if t != 'communication'})
    generator.generate(output, 'full-v3', shapes + legacy_coverage_shapes(hw.device), observations + links)
    return OperatorDataPackage.load(output)


def profile(output, executor):
    """Compact Chrome/Perfetto profile; detailed routing is reconstructible from seed/config."""
    stage_us = defaultdict(float)
    layer_us = defaultdict(float)
    with gzip.open(output, 'wt', encoding='utf-8') as stream:
        stream.write('{"displayTimeUnit":"ms","metadata":{"timing":"SIMULATED full-model decode"},"traceEvents":[')
        first = True
        def emit(row):
            nonlocal first
            if not first:
                stream.write(',')
            json.dump(row, stream, separators=(',', ':'))
            first = False
        for e in executor.trace.events:
            name = e.stage or e.operation or e.event_type
            stage_us[name] += e.service_duration * 1e6
            layer_us[e.layer_id] += e.service_duration * 1e6
            args = dict(iteration=e.iteration_id, block=e.block_id, layer=e.layer_id,
                source_id=e.source_id, payload_bytes=e.payload_bytes,
                node_id=e.node_id, predecessors=e.predecessor_ids,
                queue_us=e.queue_duration * 1e6,
                components=e.details.get('components', []))
            emit(dict(name=name, cat=e.event_type, ph='X', pid=1, tid=e.resource_id,
                ts=e.start_at * 1e6, dur=e.service_duration * 1e6, args=args))
        for e in executor.boundary_events:
            stage_us[e['name']] += e['service_duration'] * 1e6
            emit(dict(name=e['name'], cat='model-boundary', ph='X', pid=1, tid=e['resource_id'],
                ts=e['start_at'] * 1e6, dur=e['service_duration'] * 1e6, args=e))
        stream.write(']}')
    return dict(stage_us), dict(layer_us)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-root', type=Path, default=ROOT/'outputs/deepseek-mla-a800-20260916')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/deepseek-v3-full-decode-uniform-20260916')
    p.add_argument('--batches', type=int, nargs='+', default=[1, 8, 32])
    p.add_argument('--contexts', type=int, nargs='+', default=[4096])
    p.add_argument('--af-batch-sizes', type=int, nargs='+', default=[8])
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--f-gpus', type=int, default=32)
    p.add_argument('--a-gpus', type=int, default=1, help='A tensor-parallel ranks for the same requests')
    p.add_argument('--hypothetical-f-memory-gib', type=int,
        help='Diagnostic only: change F capacity, retaining A800 latency and assumed power. Not physically deployable.')
    args = p.parse_args()
    if min(*args.batches, *args.contexts, *args.af_batch_sizes, args.steps, args.f_gpus, args.a_gpus) < 1:
        p.error('Counts must be positive')
    if args.hypothetical_f_memory_gib is not None and args.hypothetical_f_memory_gib < 1:
        p.error('Hypothetical F capacity must be positive')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output/'traces').mkdir(exist_ok=True)
    hw = json.loads((args.measurement_root/'hardware-assumptions.json').read_text(encoding='utf-8'))
    hw.update(status='Mixed measured compute / theoretical missing shapes; virtual cluster',
        notes=f'{args.a_gpus+args.f_gpus}-card fabric, launch costs, peak bandwidth and fixed average power are assumptions; not a full-cluster measurement.')
    hypothetical_bytes = (args.hypothetical_f_memory_gib * 2**30
                          if args.hypothetical_f_memory_gib is not None else None)
    if hypothetical_bytes is not None:
        hw['hypothetical_f_memory_capacity_bytes'] = hypothetical_bytes
        hw['hypothetical_f_memory_note'] = ('F capacity only; A capacity, compute, bandwidth and assumed power unchanged. '
            'No weight offload or additional memory-system power modeled. Not a physical A800 deployment.')
    communication = json.loads((ROOT/'data/operator_data/af-communication-separated.json').read_text(encoding='utf-8'))
    configs = {af: configuration(True, args.a_gpus, args.f_gpus, af, hw,
        router_location='f', communication_timing=communication['host'],
        hypothetical_f_memory_bytes=hypothetical_bytes) for af in args.af_batch_sizes}
    for af, config in configs.items():
        write_json(output/f'cluster-af{af}.json', config[0])
    plan = next(iter(configs.values()))[-1]
    write_json(output/'runtime-plan.json', plan.summary())
    write_json(output/'hardware-assumptions.json', hw)
    write_json(output/'communication-model.json', communication)
    experiments = [(b,c,af) for c in args.contexts for af in args.af_batch_sizes for b in args.batches]
    discovery = RecordingLookup(hw, communication)
    for batch, context, af in experiments:
        print(f'Discover full model: B={batch}, context={context}, AF={af}', flush=True)
        simulate_full(configs[af], discovery, batch, context, args.steps, trace=False)
    shapes = list(discovery.shapes.values())
    observations, sources = load_measurements(args.measurement_root, hw['device'])
    requested = {signature(s['table'], s['key']) for s in shapes}
    observations = [o for o in observations if signature(o.table_name, o.key) in requested]
    package = make_package(output/'operator-data', shapes, hw, communication, observations, sources)
    raw_lookup = OperatorDataLookup(package)
    audit = []
    for s in shapes:
        result = raw_lookup.lookup(s['table'], s['key'])
        audit.append(dict(s, source_id=result.source_id,
            timing_method=result.values.get('timing_method'), latency_us=result.latency_us))
    write_json(output/'shape-provenance.json', audit)
    write_json(output/'measurement-backlog.json', [s for s in audit
        if s['table'] != 'communication' and s['timing_method'] != 'measured'])
    rows = []
    power_w = (args.a_gpus + args.f_gpus) * hw['device_power_w']
    for batch, context, af in experiments:
        print(f'Run full model: B={batch}, context={context}, AF={af}', flush=True)
        lookup = RecordingLookup(hw, communication, raw_lookup)
        executor, iterations = simulate_full(configs[af], lookup, batch, context, args.steps)
        total_seconds = sum(i.makespan for i in iterations)
        step_seconds = total_seconds / args.steps
        trace = f'traces/c{context}-b{batch}-af{af}.trace.json.gz'
        stage, layer = profile(output/trace, executor)
        resource_seconds, queue_seconds = defaultdict(float), defaultdict(float)
        for e in executor.trace.events:
            for resource in e.resource_ids or (e.resource_id,):
                resource_seconds[resource] += e.service_duration
                queue_seconds[resource] += e.queue_duration
        for e in executor.boundary_events:
            resource_seconds[e['resource_id']] += e['service_duration']
            queue_seconds[e['resource_id']] += e['queue_duration']
        rows.append(dict(batch=batch, context=context, af_batch_size=af,
            avg_decode_step_ms=step_seconds*1000,
            tokens_per_s_per_user=1/step_seconds, total_tokens_per_s=batch/step_seconds,
            tokens_per_s_per_watt=batch/step_seconds/power_w, total_assumed_power_w=power_w,
            trace=trace, validation='passed', layer_count=plan.total_layers,
            resource_busy_fraction={k:v/total_seconds for k,v in resource_seconds.items()},
            resource_queue_ms_sum={k:v*1000 for k,v in queue_seconds.items()},
            compute_query_counts=[dict(table=t, timing_method=m, count=n) for (t,m),n in sorted(lookup.methods.items())],
            component_service_us_sum=stage, layer_service_us_sum=layer))
        write_json(output/'results.partial.json', rows)
        print(f"{step_seconds*1000:.3f} ms/step; {1/step_seconds:.3f} tokens/s/user", flush=True)
        del executor
    limitations = [
        'CPU simulation of complete 61-layer decode plus embedding, final norm, lm_head and greedy argmax; no checkpoint inference.',
        'Prefilled KV cache; excludes prefill/TTFT, arrivals, API/CPU serving overhead and MTP/speculation.',
        'BF16 architecture, not the official FP8 implementation; seeded uniform random distinct top-8 per request/layer/step, not noaux_tc routing.',
        'Exact matching A800 runtime shapes reuse eager microbenchmarks; all other compute shapes use assumed Roofline + launch.',
        'Eager warm-weight microbenchmarks include host gaps; full-stack cache behavior and global CPU scheduling remain uncalibrated.',
        'F leader computes routing after its activation arrives; all F ranks receive full activations and explicit routing metadata.',
        'A TP splits MLA heads; Q/KV down projections and compressed KV cache replicate. Every A Combine owner receives routing metadata.',
        'A boundary uses vocab-sharded embedding/lm_head weights: conservative full-row lookup on each A rank then embedding all-reduce; replicated final norm, vocab-sharded lm_head, logits all-gather and replicated greedy argmax.',
        'All virtual GPUs share an assumed fabric; F metadata and collectives share the F-to-A lane. Not calibrated multirack networking.',
        'Fixed assumed per-card power; no CPU/network power. Weight and KV capacity enforced, workspace peaks not modeled.',
        'One fixed-batch synchronous decode iteration by default; rates are predictions, not measured serving throughput.',
        'Component/layer service sums include parallel work and must not be summed as end-to-end latency.',
    ]
    if hypothetical_bytes is not None:
        limitations.insert(0, f'HYPOTHETICAL CAPACITY: F has {args.hypothetical_f_memory_gib} GiB per worker with unchanged A800 compute/bandwidth and assumed 250 W power. '
            'A retains its original capacity. Capacity checks remain active. No offload; not deployable on physical A800s or a real-system energy-efficiency prediction.')
    write_json(output/'summary.json', dict(status='full-model-decode-simulation-mixed-calibration',
        model='DeepSeek-V3 671B BF16', layers=61, dense_layers=3, moe_layers=58,
        virtual_a_gpus=args.a_gpus, virtual_f_gpus=args.f_gpus, router='F leader', combine='A',
        hypothetical_f_memory_gib=args.hypothetical_f_memory_gib,
        synthetic_routing='uniform random without replacement; seed 42; independent of AF block boundaries',
        steps=args.steps, rows=rows, limitations=limitations))
    plot_results(output, rows, args.f_gpus, a_gpus=args.a_gpus,
                 hypothetical_f_memory_gib=args.hypothetical_f_memory_gib)
    (output/'README.md').write_text('# Full DeepSeek-V3 decode simulation\n\n'+
        '\n'.join('- '+s for s in limitations)+'\n\n'+
        'See summary.json, curves.png, shape-provenance.json, measurement-backlog.json, '+
        'runtime-plan.json and gzip-compressed Chrome/Perfetto traces/.\n', encoding='utf-8')
    print(f'Completed: {output}', flush=True)


def plot_results(output, rows, f_gpus, *, a_gpus=1, hypothetical_f_memory_gib=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    reference = max(rows, key=lambda r: r['tokens_per_s_per_watt'])
    baseline_tps_per_mw = reference['tokens_per_s_per_watt'] * 1e6
    write_json(output/'plot-axis-metadata.json', dict(
        x_label='Interactivity (TPS / User)', y_label='Throughput (TPS / MW)',
        y_display='Relative multiples of one shared TokenSim reference, like NVIDIA Figure 11',
        baseline_tps_per_mw=baseline_tps_per_mw,
        baseline_case={k: reference[k] for k in ('batch', 'context', 'af_batch_size')},
        baseline_scope='Maximum TPS/MW among all points in this figure; not NVIDIA baseline',
        point_count=len(rows),
        hypothetical_f_memory_gib=hypothetical_f_memory_gib,
        normalization='y = total_tokens_per_s / (total_assumed_power_w / 1e6) / baseline_tps_per_mw'))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for context, af in sorted({(r['context'],r['af_batch_size']) for r in rows}):
        selected = sorted([r for r in rows if (r['context'],r['af_batch_size']) == (context,af)], key=lambda r:r['batch'])
        line, = ax.plot([r['tokens_per_s_per_user'] for r in selected],
            [r['tokens_per_s_per_watt'] * 1e6 / baseline_tps_per_mw for r in selected], 'o-', label=f'Context {context//1024}K; AF cap {af}')
        label_batches = ({1, 2, 4, 8, 12, 16, 24, 32, 64, 128,
                          selected[0]['batch'], selected[-1]['batch']}
                         if len(selected) > 10 else {r['batch'] for r in selected})
        if hypothetical_f_memory_gib is not None and len(selected) > 10:
            label_batches = {1, 2, 4, 8, 12, 16, 32, selected[-1]['batch']}
        for r in selected:
            if r['batch'] not in label_batches:
                continue
            ax.annotate(str(r['batch']), (r['tokens_per_s_per_user'],r['tokens_per_s_per_watt'] * 1e6 / baseline_tps_per_mw),
                xytext=(5,5), textcoords='offset points', color=line.get_color())
    ax.set(xlabel='Interactivity\n(TPS / User)',
        ylabel='Throughput\n(TPS / MW)',
        xlim=(0, max(r['tokens_per_s_per_user'] for r in rows) * 1.08), ylim=(0, 1.08),
        title=f'DeepSeek-V3 61 layers + input/output | virtual {a_gpus}A{f_gpus}F A800' +
        (f'\nHYPOTHETICAL F memory: {hypothetical_f_memory_gib} GiB; not physically deployable'
         if hypothetical_f_memory_gib is not None else ''))
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:.1f}x'))
    ax.grid(alpha=.2)
    ax.legend()
    fig.text(.5,.02,
        f'1.0x = {baseline_tps_per_mw:,.1f} TPS/MW (TokenSim maximum in this figure; not NVIDIA baseline).\n'
        f'{len(rows)} simulated points; assumed fabric/power. Labels = selected concurrent users.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1))
    fig.savefig(output/'curves.png')
    fig.savefig(output/'curves.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
