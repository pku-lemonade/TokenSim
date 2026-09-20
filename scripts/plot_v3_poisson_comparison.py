"""Combine completed Poisson pilot results without conflating QPS and TPS/user."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/v3-poisson-comparison-20260920'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    validation = []
    history = []
    for a in (2,4,8):
        folder = ROOT/f'outputs/v3-poisson-{a}a32f-20260920'
        series = json.loads((folder/'results.json').read_text())
        assert len(series) == 7
        first_request = json.loads((folder/'qps-1/requests.json').read_text())[0]
        previous = ROOT/f'outputs/v3-virtual-Na32f-20260919/a{a}/summary.json'
        if previous.exists():
            reference = next(r for r in json.loads(previous.read_text(encoding='utf-8'))['results'] if r['batch_size']==1)
            difference_us = abs(first_request['service_ms']*1000-reference['avg_step_us'])
            assert difference_us < 1e-4
            history.append(dict(a_tp=a, batch_size=1, difference_us=difference_us, status='passed'))
        for row in series:
            case = folder/f"qps-{row['qps']:g}"
            checks = json.loads((case/'validation.json').read_text())
            assert checks['status'] == 'passed'
            requests = json.loads((case/'requests.json').read_text())
            assert len(requests) == 64
            assert all(abs(r['queue_ms']+r['service_ms']-r['latency_ms']) < 1e-6 for r in requests)
            row = dict(a_tp=a, f_ep=32, **row, profile=str(case/'operators.trace.json.gz'))
            rows.append(row)
            validation.append(dict(a_tp=a, qps=row['qps'], **checks))
    with (OUT/'results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (OUT/'validation.json').write_text(json.dumps(validation, indent=2))
    (OUT/'history-regression.json').write_text(json.dumps(history, indent=2))
    sources = ['scripts/run_v3_poisson.py', 'scripts/deepseek_v3_runtime.py', 'TokenSim/af/executor.py',
        'TokenSim/af/planning.py', 'TokenSim/af/operator_provider.py', 'TokenSim/af/runtime.py']
    (OUT/'source-hashes.json').write_text(json.dumps({f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(12,8), dpi=160)
    definitions = [('mean_latency_ms','p95_latency_ms','Latency (ms/request)','Total latency'),
        ('mean_queue_ms','p95_queue_ms','Scheduler waiting time (ms/request)','Queue before batch admission'),
        ('completed_qps_during_arrivals','completed_qps_including_drain','Completed requests/s (=tokens/s)','Delivered throughput'),
        ('backlog_at_last_arrival','max_pending','Requests','Unfinished requests and queue')]
    for ax, (first, second, ylabel, title) in zip(axes.flat, definitions):
        for a, color, marker in zip((2,4,8),('tab:blue','tab:orange','tab:green'),('o','s','^')):
            subset = [r for r in rows if r['a_tp']==a]
            ax.plot([r['qps'] for r in subset], [r[first] for r in subset], marker+'-', color=color, label=f'{a}A32F')
            ax.plot([r['qps'] for r in subset], [r[second] for r in subset], '--', color=color, alpha=.7)
        ax.set(xscale='log',xlabel='Offered Poisson arrival rate (requests/s)',ylabel=ylabel,title=title)
        ax.set_xticks([1,4,8,16,32,64,128], ['1','4','8','16','32','64','128'])
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ax, note in zip(axes.flat, ('Solid: mean; dashed: P95','Solid: mean; dashed: P95',
        'Solid: during arrivals; dashed: includes drain','Solid: unfinished at last arrival; dashed: peak pending')):
        ax.text(.02,.98,note,transform=ax.transAxes,ha='left',va='top',fontsize=8)
    fig.suptitle('DeepSeek-V3 671B | Poisson QPS sweep | A=2,4,8; F=32')
    fig.text(.5,.014,'64 requests/rate, seed 42; 2048 cached tokens + 1 decode token; FIFO batch cap 32, AF block 8.\n'
        'A100 SXM / NVLink3 theoretical simulation on CPU; finite pilot, no prefill or steady-state guarantee.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.97))
    fig.savefig(OUT/'poisson-qps-comparison.png'); fig.savefig(OUT/'poisson-qps-comparison.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5), dpi=180)
    for a, color, marker in zip((2,4,8),('tab:blue','tab:orange','tab:green'),('o','s','^')):
        subset = [r for r in rows if r['a_tp']==a]
        ax.plot([r['qps'] for r in subset],
            [r['completed_qps_including_drain'] for r in subset],
            marker+'-', color=color, label=f'{a}A32F', linewidth=2)
    ax.set(xscale='log', xlabel='Offered Poisson arrival rate (requests/s)',
        ylabel='Throughput (tokens/s)', title='DeepSeek-V3 671B: throughput vs. offered QPS', ylim=(0,None))
    ax.set_xticks([1,4,8,16,32,64,128], ['1','4','8','16','32','64','128'])
    ax.grid(alpha=.2); ax.legend()
    fig.text(.5,.015,'Total output tokens / time from simulation start to final completion (includes drain).\n'
        '64 requests/rate; one output token/request; no power normalization; theoretical CPU simulation.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.085,1,1))
    fig.savefig(OUT/'throughput-vs-qps.png'); fig.savefig(OUT/'throughput-vs-qps.svg')
    plt.close(fig)
    text = ['# V3 Poisson arrival pilot','',
        'Full 61-layer DeepSeek-V3 671B BF16; A TP=2/4/8, F EP=32, expert TP=1.',
        'A100 SXM 80GB + NVLink3; each F worker has virtual 1536 GiB capacity only.',
        'Pure analytical operator timing; executed on local CPU. Router A.',
        'Poisson rates 1/4/8/16/32/64/128 requests/s. 64 arrivals per rate, seed 42.',
        'QPS is the exponential arrival generator parameter. Realized rate differs for a finite sample; it is exported separately.',
        'Each request starts with 2048 cached tokens and generates exactly one token; no prefill.',
        'FIFO: admit up to 32 queued requests when the prior iteration ends; AF block=8; no batching timer.',
        'Arrivals continue during execution. All requests drain after the final arrival.',
        'Per-request completion follows its AF block, not the last block in the batch.',
        'Queue latency is arrival-to-admission. Service latency includes resource contention after admission.',
        'The final backlog includes in-flight requests; peak pending counts only requests waiting for admission.',
        'Throughput uses t=0 to last arrival or t=0 to final completion, as labeled.',
        'No reciprocal latency is mislabeled as steady-state TPS/user. No whole-model GPU measurements.',
        'One finite seed and 64 requests are a pilot, not a stable capacity estimate or steady-state P95.',
        'results.csv links each compressed Chrome profiling trace; validation.json records all dependency checks.',
        '', '| A | QPS | Mean latency ms | P95 ms | Mean queue ms | Backlog at last arrival |',
        '|---|---|---|---|---|---|']
    for r in rows:
        text.append(f"| {r['a_tp']} | {r['qps']:g} | {r['mean_latency_ms']:.2f} | {r['p95_latency_ms']:.2f} | {r['mean_queue_ms']:.2f} | {r['backlog_at_last_arrival']} |")
    (OUT/'README.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
