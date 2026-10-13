"""Uncalibrated whole-model TokenSim baseline for the server's tiny Qwen MoE."""
import argparse
import json
import hashlib
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.af.timeline import summarize_timeline, summarize_message_waits
from TokenSim.af.calibration import HOST_COVERAGE, runtime_fingerprint
from TokenSim.config.model_config import ModelSpec
from TokenSim.hardware.device.sourced_value import SourcedValue
from scripts.deepseek_v3_runtime import configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--protocol', choices=['partial_return', 'vllm_afd'], default='vllm_afd')
    parser.add_argument('--native-control', action=argparse.BooleanOptionalAction, default=True,
                        help='Model eager native control messages; disable only to reproduce the older baseline')
    parser.add_argument('--native-call-order', action='store_true')
    parser.add_argument('--pkua100-local-domains', action='store_true',
                        help='Explicit same-node GPU2/3 placement for current hierarchical runtime')
    parser.add_argument('--host-profile', type=Path)
    parser.add_argument('--operation-profile', type=Path)
    parser.add_argument('--transport-profile', type=Path)
    parser.add_argument('--device-service-profile', type=Path)
    parser.add_argument('--device-service-basis', type=float,
                        help='Uncalibrated device coefficient basis for fitting pure GPU services only')
    parser.add_argument('--routing-replay', type=Path)
    parser.add_argument('--calibration-environment', type=Path)
    parser.add_argument('--describe-runtime', action='store_true', help='Write runtime fingerprint without simulation')
    for name in ('a-step-prepare-us', 'f-step-prepare-us', 'compute-submit-us', 'host-receive-submit-us',
                 'host-submit-us', 'host-completion-us',
                 'control-encode-us', 'control-h2d-us', 'control-length-readback-us',
                 'control-payload-readback-us', 'control-decode-us'):
        parser.add_argument('--' + name, type=float, default=None,
                            help='Uncalibrated additional host service, excluding existing kernel launch costs')
    args = parser.parse_args()
    if any(batch < 1 or batch > 32 for batch in args.batches):
        parser.error('native comparison batch must be between 1 and server max_num_seqs=32')
    model = ModelSpec.from_mapping(dict(
        model_id='qwen3-moe-tiny-random', hidden_size=256, intermediate_size=512,
        num_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        vocab_size=128, max_position_embeddings=256, attention_type='gqa',
        dtype='bf16', kv_cache_dtype='bf16',
        moe=dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=256,
                 num_shared_experts=0, num_moe_layers=2)))
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for batch in args.batches:
        # Match the native engine's fixed max_num_seqs rather than changing
        # memory reservation/runtime identity for every calibration batch.
        base = configuration(a_tp=1, f_tp=1, af_batch_size=32 if args.protocol == 'vllm_afd' else batch)
        device_id = 'alignment-a100-pcie-80g'
        device = replace(base.hardware.device('a100_sxm_80g'), device_id=device_id,
                         display_name='A100 PCIe 80GB alignment assumption', aliases=(),
                         memory_bandwidth_bytes_per_s=SourcedValue(1935e9, 'alignment-spec-assumption', 'D'))
        if args.device_service_basis is not None:
            if args.device_service_profile or not math.isfinite(args.device_service_basis) or args.device_service_basis<0:
                parser.error('invalid or conflicting device service basis')
            device=replace(device,analytical={**device.analytical,'kernel_launch_us':args.device_service_basis})
        if args.device_service_profile:
            raw=args.device_service_profile.read_bytes();cal=json.loads(raw)
            if (cal.get('timing_scope')!='device_service' or cal.get('contains_cpu_submit') is not False
                or cal.get('contains_wait') is not False or cal.get('model_id')!=model.model_id
                or cal.get('unit')!='us' or cal.get('training_batches')!=[1,8]
                or type(cal['kernel_fixed_us']) not in (int,float)
                or not math.isfinite(cal['kernel_fixed_us']) or cal['kernel_fixed_us']<0):
                parser.error('invalid device-only calibration')
            device=replace(device,analytical={**device.analytical,'kernel_launch_us':cal['kernel_fixed_us']},
                analytical_source_id='device_service:'+hashlib.sha256(raw).hexdigest())
        base.hardware.devices.add(device)
        doc = base.doc
        if args.pkua100_local_domains:
            doc['af']['communication_domains'] = dict(
                node_size=8, node_count=1, placement_strategy='manual',
                device_indices={'a-pool:0': 2, 'f-pool:0': 3},
                local_bandwidth_bytes_per_s=300e9, local_latency_us=1.0,
                source_id='pkua100-gpu23-nv12:bandwidth-and-alpha-catalog-assumptions')
        for group in doc['worker_groups']:
            group['hardware'] = device_id
        doc['af'].update(router_location='f', transport_router_logits=False,
                         execution_protocol=args.protocol, phases=['decode'],
                         communication_timing=dict(enabled=True,
                             native_control_enabled=args.native_control and args.protocol == 'vllm_afd',
                             native_call_order_enabled=args.native_call_order,
                             **{name: value for name, value in vars(args).items()
                                if name.endswith('_us') and value is not None}))
        for argument, field in (('host_profile', 'engine_overhead_profile'),
                                ('transport_profile', 'transport_service_profile'),
                                ('operation_profile', 'operation_service_profile'),
                                ('routing_replay', 'routing_replay'),
                                ('calibration_environment', 'calibration_environment')):
            path = getattr(args, argument)
            if path is not None:
                doc['af'][field] = json.loads(path.read_text(encoding='utf-8'))
        for entry in doc['af']['links'] + doc['af']['execution_domains']:
            entry['phases'] = ['decode']
        runtime = build_model_runtime(doc, hardware=base.hardware, model=model)
        fingerprint = runtime_fingerprint(runtime.plan, runtime.cluster, runtime.workers, runtime.hardware)
        if args.describe_runtime:
            (args.output / f'runtime-b{batch}.json').write_text(json.dumps(dict(
                runtime_fingerprint=fingerprint, protocol=args.protocol, batch=batch), indent=2))
            continue
        executor, rows, _, _ = simulate_decode(runtime, batch, 129, steps=31,
                                               fallback='analytical_only')
        events = [e.as_dict() for e in executor.trace.events]
        stages = defaultdict(float)
        payloads = defaultdict(int)
        control_payloads = defaultdict(int)
        compute_by_endpoint = defaultdict(float)
        queue_us = readiness_wait_us = 0.0
        for e in events:
            label = e['stage'] or e['operation'] or e['event_type']
            stages[label] += e['service_duration'] * 1e6 / 31
            queue_us += e['queue_duration'] * 1e6 / 31
            readiness_wait_us += e.get('details', {}).get('readiness_wait_duration', 0) * 1e6 / 31
            if e['event_type'] == 'compute':
                compute_by_endpoint[','.join(e['physical_endpoints'])] += e['service_duration'] * 1e6 / 31
            if e['event_type'] == 'transfer':
                payloads[e['direction']] += e.get('payload_bytes') or 0
            if e['event_type'] == 'control_transfer':
                control_payloads[e['operation']] += e.get('payload_bytes') or 0
        decode = dict(mean_step_us=sum(r['makespan_us'] for r in rows) / len(rows),
                      stage_service_us=dict(stages),
                      compute_service_by_endpoint_us=dict(compute_by_endpoint),
                      summed_event_queue_us=queue_us,
                      summed_readiness_wait_us=readiness_wait_us,
                      waiting_scope='event sums, not critical path; native readiness not replayed',
                      mean_payload_bytes={k: v / 31 for k, v in payloads.items()})
        decode['mean_control_payload_bytes'] = {k: v / 31 for k, v in control_payloads.items()}
        critical = defaultdict(float)
        step_start = 0.0
        timeline_sample = None
        communication_windows = []
        for row in rows:
            communication_windows.extend(summarize_message_waits(executor.trace.events_for(row['iteration_id'])))
            step_end = step_start + row['makespan_us'] / 1e6
            timeline = summarize_timeline(executor.trace.events_for(row['iteration_id']),
                step_start, step_end, executor.resources.resource_capacities())
            for name, duration in timeline['critical_duration_by_category'].items():
                critical[name] += duration * 1e6 / len(rows)
            if timeline_sample is None:
                timeline_sample = timeline
            step_start = step_end
        decode['mean_critical_chain_us'] = dict(critical)
        decode['first_step_timeline_seconds'] = timeline_sample
        decode['communication_windows_seconds'] = communication_windows
        decode['communication_windows_scope'] = 'per-observer alternatives; do not sum endpoints or messages as step wall time'
        result = dict(batch=batch, input_tokens=128, output_tokens=32,
                      communication_domains=doc['af'].get('communication_domains'),
                      runtime_fingerprint=fingerprint,
                      host_profile=(executor.overhead_profile.describe() if executor.overhead_profile else None),
                      operation_profile=(executor.operation_profile.describe() if executor.operation_profile else None),
                      routing_replay=(executor.routing_replay.describe() if executor.routing_replay else None),
                      control_protocol_status=('modeled_uncalibrated' if doc['af']['communication_timing']['native_control_enabled'] else 'disabled_baseline'),
                      fixed_host_assumptions=doc['af']['communication_timing'],
                      effective_host_costs={name: executor.host_cost_provenance(name, tokens=batch)
                                            for name in sorted(set(HOST_COVERAGE) | (set(executor.overhead_profile.document['costs'])
                                                if executor.overhead_profile is not None else set()))},
                      decode=decode, execution_protocol=args.protocol, calibration_status='uncalibrated',
                      prefill_scope='excluded; context is an input, no prefill handoff validated',
                      fallback='analytical_only',
                      limitations=['synthetic routing, not captured plugin assignments',
                                   'control torch.distributed and data PyNccl services remain separately uncalibrated',
                                   'fixed host preparation/dispatch is an approximation; native call sequence not replayed',
                                   'Q/K norm modeled; native fusion/launch costs remain uncalibrated',
                                   'catalog NVLink service is uncalibrated'])
        if executor.routing_replay is not None:
            result['limitations'][0] = 'routing assignments replayed; capture provenance and timing still require independent verification'
        if executor.overhead_profile is not None:
            result['calibration_status'] = ('partial_host_calibration' if
                executor.overhead_profile.document['evidence']['kind'] == 'measured'
                else 'synthetic_test_profile_not_calibration')
        if executor.operation_profile is not None:
            result['calibration_status'] = ('partial_operation_calibration' if
                executor.operation_profile.describe()['evidence_kind'] == 'measured' else 'synthetic_operation_profile')
        if args.native_call_order:
            result['limitations'][1]='native message call ordering modeled; CPU/kernel interleaving remains a call-group approximation'
        if args.device_service_profile:
            result['device_service_profile_sha256']=hashlib.sha256(args.device_service_profile.read_bytes()).hexdigest()
            result['limitations'][2]='GPU-only fixed service fitted to classified kernel sums; aggregate approximation'
        if args.transport_profile:
            result['transport_profile_sha256']=hashlib.sha256(args.transport_profile.read_bytes()).hexdigest()
            result['limitations'][3]='paired device transport service; cross-device clock and profiler uncertainty retained'
        if executor.overhead_profile and executor.overhead_profile.document['schema_version']==4:
            result['calibration_status']='scoped_profiles_loaded_validation_report_required'
        (args.output / f'tokensim-b{batch}.json').write_text(json.dumps(result, indent=2))
        (args.output / f'tokensim-b{batch}-decode-events.json').write_text(json.dumps(events))
        summaries.append(result)
        print(json.dumps({'batch': batch, 'decode_us': decode['mean_step_us'],
                          'protocol': args.protocol}))
    (args.output / 'tokensim-summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
