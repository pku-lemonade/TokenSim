"""Offline TokenSim curves with measured compute and separated assumed communication.

End-to-end points and timelines are simulated, not native two-GPU measurements.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_deepseek_mla import (ShapeDiscovery, configuration, simulate,
    save_trace, legacy_coverage_shapes, write_json)
from TokenSim.operator_data import (Calibration, EvidenceGrade, HardwareProfile,
    Observation, OperatorDataGenerator, OperatorDataLookup, OperatorDataPackage,
    SourceRecord, estimate_operator)


def signature(row):
    return row['table'], json.dumps(row['key'], sort_keys=True)


def positive_fit(x, y):
    """Least squares y = intercept + slope*x with nonnegative intercept/slope."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    assert len(x) >= 2 and np.ptp(x) > 0 and np.all(y > 0)
    a, b = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0]
    candidates = [(max(float(y.mean()-1e-12*x.mean()), 0), 1e-12),
                  (0, max(float(x@y/(x@x)), 1e-12))]
    if a >= 0 and b > 0:
        candidates.append((float(a), float(b)))
    a, b = min(candidates, key=lambda pair: float(np.sum((pair[0]+pair[1]*x-y)**2)))
    predicted = a+b*x
    errors = np.abs(predicted-y)/y
    return dict(intercept=a, slope=b, mape=float(errors.mean()), max_relative_error=float(errors.max()),
                predicted=predicted.tolist(), slope_at_lower_bound=b <= 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--power-w', type=float, default=250.)
    parser.add_argument('--communication-model', type=Path,
        default=Path(__file__).resolve().parents[1]/'data/operator_data/af-communication-separated.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.power_w <= 0:
        parser.error('power-w must be positive')
    root = args.root
    output = args.output or root/'simulation-separated'
    communication_model = json.loads(args.communication_model.read_text())
    if communication_model['host']['enabled'] is not True:
        raise ValueError('This experiment requires separated communication timing')
    if communication_model['transport']['timing_scope'] != 'transport_service':
        raise ValueError('Transport profile must exclude submission, synchronization and queueing')
    output.mkdir(parents=True, exist_ok=True)
    (output/'traces').mkdir(exist_ok=True)
    hw = json.loads((root/'hardware-assumptions.json').read_text())
    hw.update(device_power_w=args.power_w, power_model_type='fixed_assumed_average',
        status='Exact measured compute shapes; assumed separate host and transport costs; CPU simulated layer schedule',
        notes='Roofline peaks remain assumptions used only for holdout formula analysis and unused schema seeds. Runtime compute uses exact target-device observations. Power is an assumed average, not TDP or measured draw.')
    hardware = HardwareProfile(**{f.name: hw[f.name] for f in fields(HardwareProfile) if f.name in hw})
    sources, observations, measurements, contracts = [], [], [], {}
    for role in ('A', 'F'):
        directory = root/('compute-'+role)
        contract = json.loads((directory/'measurement-contract.json').read_text())
        rows = json.loads((directory/'measurements.json').read_text())
        expected = json.loads((directory/'cases.json').read_text())
        assert contract.get('failed') == 0 and contract.get('completed') == len(expected)
        assert len(rows) == len(expected) and all(r['status'] == 'ok' and r['validation']['passed'] for r in rows)
        assert contract['input_sha256'] == hashlib.sha256((directory/'cases.json').read_bytes()).hexdigest()
        source_id = 'measured:a800:'+role
        sources.append(SourceRecord(source_id, EvidenceGrade.A, 'measured', hw['device'], 'analytical',
            str(directory/'measurement-contract.json'), hashlib.sha256((directory/'measurements.json').read_bytes()).hexdigest()))
        for row in rows:
            assert row['key']['device'] == hw['device']
            observations.append(Observation(row['table'], row['key'], row['device_interval_us'], source_id, row['split']))
        measurements.extend(rows)
        contracts[role] = contract
    discovery = ShapeDiscovery(hw)
    experiments = json.loads((root/'experiments.json').read_text())
    for item in experiments:
        simulate(discovery, configuration(af_batch_size=item['af_batch_size'], assumptions=hw, router_location='f'), item['batch'], item['context'])
    shapes = list(discovery.shapes.values())
    measured_keys = {signature(r) for r in measurements if r['split'] != 'holdout'}
    missing = [s for s in shapes if s['table'] != 'communication' and signature(s) not in measured_keys]
    if missing:
        raise RuntimeError('Missing exact measured runtime shapes: '+str(missing))

    mla = [r for r in measurements if r['table'] == 'mla_attention']
    train = [r for r in mla if r['split'] != 'holdout']
    fitted = positive_fit([estimate_operator('mla_attention', r['key'], hardware).latency_us for r in train],
                          [r['device_interval_us'] for r in train])
    calibration = Calibration(multiplier=fitted['slope'], offset_us=fitted['intercept'],
                              calibration_id='A800-MLA-runtime-only-fit-holdouts-excluded')
    holdouts = []
    for r in mla:
        if r['split'] != 'holdout':
            continue
        predicted = estimate_operator('mla_attention', r['key'], hardware, calibration).latency_us
        actual = r['device_interval_us']
        holdouts.append(dict(case_id=r['case_id'], key=r['key'], actual_us=actual, predicted_us=predicted,
                             relative_error=abs(predicted-actual)/actual))
    write_json(output/'mla-holdout.json', dict(fit=fitted, holdouts=holdouts,
        note='Holdouts were not fitted or used as runtime observations. These predictions do not determine the plotted exact-shape curve. Intercept includes eager gaps; it is not isolated GPU launch latency.'))

    comm = json.loads((root/'communication/measurements.json').read_text())
    assert json.loads((root/'communication/done.json').read_text())['status'] == 'ok'
    fits = {}
    for direction in ('a_to_f', 'f_to_a'):
        points = [r for r in comm if r['mode'] == direction]
        assert len(points) >= 6 and all(r['correctness'] for r in points)
        fit = positive_fit([r['payload_bytes']/1e6 for r in points], [r['device_interval_us'] for r in points])
        fit['effective_bandwidth_bytes_per_s'] = 1e12/fit['slope']
        fit['points'] = [dict(payload_bytes=r['payload_bytes'], measured_us=r['device_interval_us']) for r in points]
        residual_rms = float(np.sqrt(np.mean((np.array(fit['predicted']) -
                          np.array([r['device_interval_us'] for r in points]))**2)))
        fit['residual_rms_us'] = residual_rms
        fit['size_dependent_span_us'] = float(np.ptp(fit['predicted']))
        fit['bandwidth_identifiable'] = (not fit['slope_at_lower_bound'] and
            fit['size_dependent_span_us'] > 3*residual_rms)
        fit['note'] = ('The fitted bandwidth is only an empirical coefficient over sampled payloads, '
            'not NVLink hardware bandwidth. Identifiability requires the fitted size effect to exceed '
            'three times residual RMS; eager submission overhead dominates these measurements.')
        fits[direction] = fit
    source_id = 'assumed:separate-transport'
    sources.append(SourceRecord(source_id, EvidenceGrade.D, 'assumption', hw['device'], 'analytical',
        'Transport-only configurable assumptions; eager P2P intervals are excluded from runtime calibration',
        hashlib.sha256(args.communication_model.read_bytes()).hexdigest()))
    for shape in shapes:
        if shape['table'] == 'communication':
            transport = communication_model['transport'][shape['key']['direction']]
            observations.append(Observation('communication', dict(shape['key'],
                timing_scope='transport_service', **transport),
                0, source_id))
    write_json(output/'excluded-eager-communication-fit.json', dict(
        excluded_from_runtime=True, reason='Includes host submission and waiting; not transport service', fits=fits))
    write_json(output/'communication-model.json', communication_model)
    duplex = []
    for r in comm:
        if r['mode'] != 'bidirectional':
            continue
        singles = [v['device_interval_us'] for v in comm
                   if v['payload_bytes'] == r['payload_bytes'] and v['mode'] != 'bidirectional']
        duplex.append(dict(payload_bytes=r['payload_bytes'], concurrent_us=r['device_interval_us'],
                           isolated_max_us=max(singles), ratio=r['device_interval_us']/max(singles)))
    median_duplex_ratio = float(np.median([r['ratio'] for r in duplex]))
    write_json(output/'duplex-diagnostic.json', dict(points=duplex, median_ratio=median_duplex_ratio,
        note='Ratios above one show this single-host eager benchmark did not achieve ideal overlap. '
             'They do not establish hardware half-duplex; host submission can be the limit. '
             'The TokenSim curve still assumes independent full-duplex lanes and is not native end-to-end validation.'))
    generator = OperatorDataGenerator('deepseek-mla-a800-separated-communication-v2', {hw['device']:hardware}, sources,
                                     calibrations={'mla_attention':calibration})
    generator.generate(output/'operator-data', 'a800-single-layer', shapes+legacy_coverage_shapes(hw['device']),
                       observations, holdout_comparisons={'mla_attention':holdouts})
    lookup = OperatorDataLookup(OperatorDataPackage.load(output/'operator-data'))
    rows, timelines = [], {}
    for item in experiments:
        config = configuration(af_batch_size=item['af_batch_size'], assumptions=hw, router_location='f',
                               communication_timing=communication_model['host'])
        executor, iterations = simulate(lookup, config, item['batch'], item['context'])
        latency_us = iterations[0]['makespan_us']
        rate = 1e6/latency_us
        name = f"{item['kind']}-c{item['context']}-b{item['batch']}-af{item['af_batch_size']}"
        components = save_trace(output/'traces'/f'{name}.trace.json', executor)
        events = [e.as_dict() for e in executor.trace.events]
        write_json(output/'traces'/f'{name}.events.json', events)
        stages = defaultdict(float)
        for event in events:
            stage = (event['event_type'] if event['event_type'] in ('host_submit','synchronize')
                     else event['stage'] or event['direction'])
            stages[stage] += event['service_duration']*1e6
        row = dict(item, latency_ms=latency_us/1000, single_layer_user_rate=rate,
            single_layer_total_rate=item['batch']*rate,
            single_layer_rate_per_mw=item['batch']*rate/(2*args.power_w/1e6),
            total_assumed_power_w=2*args.power_w, stage_service_us=dict(stages), components_us=components,
            trace=f'traces/{name}.trace.json')
        rows.append(row)
        timelines[name] = events
    limitations = [
        'TokenSim end-to-end single-layer simulation, not a native two-GPU layer or full 671B inference measurement.',
        'BF16 synthetic uniform routing; no V3 noaux_tc selection or checkpoint/model quality test.',
        'Compute keys are exact measured observations; F expert kernels measured on GPU1, other kernels on GPU0 and reused across identical-SKU cards.',
        'Communication host submission and transport are separate resources. Their configured costs are assumptions, not newly measured calibration.',
        'Queue and dependency waiting arise from scheduling and are never inserted as fixed latency. Completion callback is zero unless configured.',
        f'Old eager P2P data is retained only for diagnostics ({median_duplex_ratio:.2f}x duplex ratio); it does not set new communication costs.',
        'Full-duplex lanes and independent host submission workers are assumptions, not native two-GPU validation.',
        'Eager device intervals include host gaps; composition does not reproduce one global CPU launch thread.',
        'Default grouped MoE kernel configs are not tuned for A800.',
        'All rates are single-layer equivalents, not complete-model generation TPS.',
        f'Power is an assumed {args.power_w:g} W/card, not measured power or rated TDP.',
    ]
    write_json(output/'summary.json', dict(status='measured-compute-assumed-separated-communication',
        model='DeepSeek-V3-shaped one MoE layer', device=hw['device'], router='F', gpu_contracts=contracts,
        communication_model=communication_model,
        cases=rows, exact_runtime_shapes=dict(Counter(s['table'] for s in shapes if s['table'] != 'communication')),
        limitations=limitations))
    write_json(output/'hardware-assumptions.json', hw)
    plt.rcParams.update({'figure.dpi':160, 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {4096:'#247ba0', 16384:'#d9683a'}
    for context in (4096, 16384):
        pipeline = [r for r in rows if r['kind'] == 'pipeline' and r['context'] == context]
        axes[0].plot([r['af_batch_size'] for r in pipeline], [r['latency_ms'] for r in pipeline],
                     'o-', color=colors[context], label=f'Context {context//1024}K')
        curve = [r for r in rows if r['kind'] == 'concurrency' and r['context'] == context]
        axes[1].plot([r['single_layer_user_rate'] for r in curve],
                     [r['single_layer_rate_per_mw']/1e6 for r in curve], 'o-', color=colors[context],
                     label=f'Context {context//1024}K')
        for r in curve if context == 4096 else []:
            axes[1].annotate(str(r['batch']), (r['single_layer_user_rate'], r['single_layer_rate_per_mw']/1e6),
                             xytext=(4, 3), textcoords='offset points', fontsize=8, color=colors[context])
    axes[0].set(xscale='log', xlabel='AF microbatch size (tokens/microbatch)', ylabel='Simulated whole-batch layer latency (ms)',
                title='Pipeline sweep (total batch = 32 tokens)')
    axes[0].set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
    axes[1].set(xlabel='Single-layer equivalent user rate\n(tokens/s/user)',
                ylabel='Single-layer equivalent throughput per power\n(10^6 tokens/s/MW)', title='Throughput vs interactivity (AF cap = 8 tokens)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('TokenSim | DeepSeek-V3 one MoE layer | 1A1F A800 80GB', fontsize=13)
    fig.text(.5, .015, f'Measured compute; separate assumed communication. CPU submit {communication_model["host"]["host_submit_us"]:g} us/message. Labels = concurrency; {2*args.power_w:g} W assumed.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(output/'curves.png')
    fig.savefig(output/'curves.svg')
    plt.close(fig)

    pipeline = [r for r in rows if r['kind'] == 'pipeline' and r['context'] == 4096]
    best = min(pipeline, key=lambda r:r['latency_ms'])
    selected = [next(r for r in pipeline if r['af_batch_size'] == 32), best]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, row in zip(axes, selected):
        name = f"pipeline-c4096-b32-af{row['af_batch_size']}"
        lanes = sorted({e['resource_id'] for e in timelines[name]})
        for e in timelines[name]:
            ax.broken_barh([(e['start_at']*1000, e['service_duration']*1000)],
                (lanes.index(e['resource_id'])-.35, .7),
                facecolors=plt.get_cmap('tab10')(e['block_id']%10), edgecolor='white', linewidth=.2)
        lane_names = {'worker:0':'A (GPU 0)', 'worker:1':'F (GPU 1)',
                      'link:af-fabric:a_to_f':'A to F', 'link:af-fabric:f_to_a':'F to A'}
        ax.set_yticks(range(len(lanes)), [lane_names.get(lane, lane) for lane in lanes], fontsize=8)
        ax.set(xlabel='Simulated time (ms)', title=f"AF microbatch {row['af_batch_size']} | batch 32 | 4K | {row['latency_ms']:.3f} ms")
        ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'pipeline-timeline.png')
    plt.close(fig)
    (output/'README.md').write_text('# A800 TokenSim single-layer results\n\n'+
        '\n'.join('- '+s for s in limitations)+'\n\n'+
        'Runtime compute coverage: all '+str(len(measured_keys))+' measured shapes; no runtime compute Roofline fallback.\n'+
        'See curves.png, pipeline-timeline.png, summary.json, mla-holdout.json, communication-model.json, and traces/.\n', encoding='utf-8')
    print(json.dumps(dict(status='ok', experiments=len(rows), best_4k_af=best['af_batch_size'],
                          output=str(output)), indent=2))


if __name__ == '__main__':
    main()
