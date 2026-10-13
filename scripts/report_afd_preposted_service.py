"""Summarize paired device diagnostics separately from inclusive exchange timing."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def report(root):
    groups = defaultdict(list)
    sources = {}
    for backend in ('pynccl', 'torch'):
        for window in range(3):
            path = root/f'{backend}-w{window}-kernels.json'
            document = json.loads(path.read_text(encoding='utf-8'))
            assert document['profiler_enabled']
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert len(document['results']) == 12
            for row in document['results']:
                assert row['samples'] == 50 and row['window'] == window
                assert math.isfinite(row['p50_device_service_us']) and row['p50_device_service_us'] > 0
                groups[(backend, row['bytes'], row['sender'], row['receiver_delay_ms'])].append(row)
    device = []
    for (backend, size, sender, delay), rows in sorted(groups.items()):
        assert {r['window'] for r in rows} == {0, 1, 2}
        batch = 8 if size == 4096 else 1
        events = json.loads(Path(f'outputs/afd-preparation-replay-20261007-b{batch}/tokensim-b{batch}-decode-events.json').read_text(encoding='utf-8'))
        backend_name = 'torch_distributed_nccl' if backend == 'torch' else backend
        direction = 'a_to_f' if sender == 0 else 'f_to_a'
        model = [e['service_duration']*1e6 for e in events if e['event_type'] in ('transfer', 'control_transfer')
                 and e['payload_bytes'] == size and e['direction'] == direction
                 and e['details']['transport_backend'] == backend_name]
        median = statistics.median(r['p50_device_service_us'] for r in rows)
        predicted = statistics.mean(model) if model else None
        device.append(dict(backend=backend, bytes=size, sender=sender, delay_ms=delay,
            windows=3, samples=150, measured_median_us=median,
            window_median_range_us=[min(r['p50_device_service_us'] for r in rows), max(r['p50_device_service_us'] for r in rows)],
            max_window_p95_us=max(r['p95_device_service_us'] for r in rows),
            model_device_service_us=predicted,
            ratio=max(predicted, median)/min(predicted, median) if predicted else None,
            scope_caveat='设备服务近似对照；模型包含解析启动假设，实测排除双方kernel开始前的CPU提交与设备排队'))
    outer = []
    for backend in ('pynccl', 'torch'):
        for mode in ('unprofiled', 'profiled'):
            for window in range(3):
                for rank in (0, 1):
                    path = root/f'{backend}-w{window}-{mode}-rank{rank}.json'
                    sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    for row in json.loads(path.read_text(encoding='utf-8')):
                        # v3 has no profiler code and predates the explicit flag.
                        assert row.get('profiler_enabled', False) == (mode == 'profiled')
                        assert not row['admissible_for_pure_transport_profile']
                        samples = row['samples']
                        assert len(samples) == 50
                        assert all(s['service_upper_us'] >= s['service_lower_us'] for s in samples)
                        outer.append(dict(backend=backend, mode=mode, window=window, sender=row['sender'],
                            bytes=row['bytes'], delay_ms=row['receiver_delay_ms'], samples=len(samples),
                            median_lower_us=row['median_lower_us'], median_upper_us=row['median_upper_us'],
                            median_ready_wait_us=statistics.median(s['ready_wait_us'] for s in samples),
                            median_clock_bound_us=statistics.median(s['clock_bound_us'] for s in samples)))
    stability = []
    perturbation = []
    for backend, sizes in [('pynccl', (512, 4096)), ('torch', (8, 179))]:
        for size in sizes:
            for sender in (0, 1):
                medians = [r['measured_median_us'] for r in device if (r['backend'], r['bytes'], r['sender']) == (backend, size, sender)]
                stability.append(dict(backend=backend, bytes=size, sender=sender,
                    delay_service_max_min_ratio=max(medians)/min(medians)))
                selected = [r for r in outer if (r['backend'], r['bytes'], r['sender'], r['delay_ms']) == (backend, size, sender, 0)]
                values = {mode:statistics.median((r['median_lower_us']+r['median_upper_us'])/2 for r in selected if r['mode']==mode)
                          for mode in ('profiled', 'unprofiled')}
                perturbation.append(dict(backend=backend, bytes=size, sender=sender,
                    **values, profiled_unprofiled_ratio=values['profiled']/values['unprofiled']))
    return dict(scope='设备服务与含提交交换区间分开报告；不生成transport校准profile',
        full_alignment_passed=False, device_samples=3600, unprofiled_samples=3600,
        model_matched_order_pass=all(r['ratio'] < 10 for r in device if r['ratio'] is not None),
        unmatched_model_cases='torch反向消息为微基准控制项，原生模型仅A到F控制，倍率留空',
        delay_stability=stability, profiler_perturbation=perturbation,
        device_results=device, outer_results=outer, source_sha256=sources,
        limitations=['CUPTI跨设备时间校准不是物理线速测量；保留原始trace供复核',
                     'CPU sidecar确保接收已提交，但会改变调度；外层区间含发送提交',
                     '真实引擎精确逐消息CPU提交/数据就绪时间线仍需独立关联',
                     '时钟误差以区间报告，不用区间中点拟合固定成本'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('outputs/afd-preposted-summary-20261007-v1'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = report(a.root)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ('device_samples', 'unprofiled_samples', 'model_matched_order_pass', 'delay_stability', 'profiler_perturbation')}))


if __name__ == '__main__':
    main()
