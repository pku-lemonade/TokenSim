"""Resumable CPU-only full-V3 A/F sweep; every distinct point has a profile."""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
A_SIZES = (1, 2, 4, 8)
BATCHES = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
AF_CAPS = (1, 2, 4, 8, 16, 32)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def cases():
    return [(a, b, af) for a in A_SIZES for b in BATCHES
            for af in sorted({min(b, cap) for cap in AF_CAPS} | {b})]


def case_name(case):
    a, b, af = case
    return f'a{a}-f32-b{b}-af{af}'


def validate_case(output, case):
    a, b, af = case
    summary = json.loads((output/'summary.json').read_text())
    assert summary['virtual_a_gpus'] == a and summary['virtual_f_gpus'] == 32
    assert summary['hypothetical_f_memory_gib'] is None
    assert len(summary['rows']) == 1
    row = summary['rows'][0]
    assert (row['batch'], row['context'], row['af_batch_size']) == (b, 4096, af)
    assert row['total_assumed_power_w'] == (a+32)*250
    assert row['validation'] == 'passed' and row['layer_count'] == 61
    assert abs(row['tokens_per_s_per_user']*row['avg_decode_step_ms']-1000) < 1e-6
    assert abs(row['total_tokens_per_s']-b*row['tokens_per_s_per_user']) < 1e-6
    assert all(-1e-10 <= value <= 1+1e-8 for value in row['resource_busy_fraction'].values())
    plan = json.loads((output/'runtime-plan.json').read_text())
    assert len(plan['physical_workers']) == a+32
    assert all(s['capacity_bytes'] == 80*2**30 and s['model_bytes'] < s['capacity_bytes']
               for s in plan['physical_workers'])
    path = output/row['trace']
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        profile = json.load(stream)
    events = profile['traceEvents']
    assert 'SIMULATED' in profile['metadata']['timing']
    layers = {e['args']['layer'] for e in events if e['args'].get('layer', -1) >= 0}
    assert layers == set(range(61))
    blocks = (b+af-1)//af
    for name in ('embedding','final_norm','lm_head','greedy_argmax'):
        assert sum(e['name'] == name for e in events) == blocks*a
    if a > 1:
        for layer in (-1,-2):
            assert sum(e['cat']=='collective' and e['args'].get('layer')==layer for e in events) == blocks
    assert abs(max(e['ts']+e['dur'] for e in events)/1000-row['avg_decode_step_ms']) < 1e-6
    validation = dict(status='passed', transformer_layers=61, a_ranks=a, blocks=blocks,
        event_count=len(events), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        validation_scope='Dependency, capacity, boundary order and timing consistency; not hardware validation')
    write(output/'validation.json', validation)
    return dict(row, a_gpus=a, f_gpus=32, case=case_name(case),
                no_pipeline=(af==b), nominal_af_caps=[cap for cap in AF_CAPS if min(cap,b)==af])


def run_case(args, case):
    output = args.output/'cases'/case_name(case)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists() and (output/'validation.json').exists():
        return validate_case(output, case)
    a,b,af = case
    command = [sys.executable, str(ROOT/'scripts/run_deepseek_v3_full.py'),
        '--a-gpus',str(a),'--f-gpus','32','--batches',str(b),'--contexts','4096',
        '--af-batch-sizes',str(af),'--measurement-root',str(args.measurement_root),
        '--output',str(output)]
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (output/'run.log').open('w') as log:
        child = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        write(output/'launch.json',dict(pid=child.pid,command=command,started_at=time.time()))
        code = child.wait()
    if code:
        raise RuntimeError(f'{case_name(case)} failed ({code}); see {output}/run.log')
    return validate_case(output,case)


def pareto(rows):
    frontier=[]
    best_y=-1
    for row in sorted(rows,key=lambda r:(-r['tokens_per_s_per_user'],-r['tokens_per_s_per_watt'],r['af_batch_size'])):
        if row['tokens_per_s_per_watt'] > best_y*(1+1e-12)+1e-12:
            frontier.append(row)
            best_y=row['tokens_per_s_per_watt']
    return sorted(frontier,key=lambda r:r['tokens_per_s_per_user'])


def plot(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    baseline=max(r['tokens_per_s_per_watt']*1e6 for r in rows)
    envelope={a:pareto([r for r in rows if r['a_gpus']==a]) for a in A_SIZES}
    best={a:[max((r for r in rows if r['a_gpus']==a and r['batch']==b),
                  key=lambda r:r['total_tokens_per_s']) for b in BATCHES] for a in A_SIZES}
    reference={a:[next(r for r in rows if r['a_gpus']==a and r['batch']==b and r['no_pipeline'])
                   for b in BATCHES] for a in A_SIZES}
    for normalized in (True,False):
        fig,ax=plt.subplots(figsize=(10,6),dpi=160)
        for a,color in zip(A_SIZES,('tab:blue','tab:orange','tab:green','tab:red')):
            for data,style,label in ((envelope[a],'o-',f'{a}A32F tuned AF'),
                                     (reference[a],'--',f'{a}A32F whole-batch')):
                selected=sorted(data,key=lambda r:r['tokens_per_s_per_user'])
                ax.plot([r['tokens_per_s_per_user'] for r in selected],
                        [r['tokens_per_s_per_watt']*1e6/(baseline if normalized else 1) for r in selected],
                        style,color=color,label=label,markersize=4,linewidth=1.8,alpha=1 if style=='o-' else .6)
        ax.set(xlabel='Interactivity (TPS / User)',ylabel='Throughput (TPS / MW)',
               title='Full DeepSeek-V3 671B BF16 | 4K context | virtual A800',xlim=(0,None),ylim=(0,None))
        if normalized:
            ax.set_ylim(0,1.08)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:.1f}x'))
        ax.grid(alpha=.2)
        ax.legend(ncol=2,fontsize=8)
        footer=(f'Shared 1.0x = {baseline:,.1f} TPS/MW; not NVIDIA baseline. ' if normalized else '')
        fig.text(.5,.025,footer+'250 W/card assumed; GPU-only power.\n'
                 'Mixed measured/theoretical simulation; multi-A shapes and shared fabric are not fully calibrated.',
                 ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.09,1,1))
        stem='curves' if normalized else 'curves-absolute'
        for suffix in ('png','svg'): fig.savefig(output/f'{stem}.{suffix}')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),dpi=160)
    for a in A_SIZES:
        ax.plot(BATCHES,[x['total_tokens_per_s']/y['total_tokens_per_s'] for x,y in zip(best[a],reference[a])],
                'o-',label=f'{a}A32F')
    ax.axhline(1,color='gray',linestyle='--')
    ax.set(xscale='log',xlabel='Concurrent users (requests)',ylabel='Total throughput / whole-batch throughput',
           title='AF microbatch pipeline benefit | same model, topology and concurrency')
    ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(output/'pipeline-gain.png'); plt.close(fig)
    write(output/'plot-axis-metadata.json',dict(x='Interactivity (TPS / User)',y='Throughput (TPS / MW)',
        shared_normalization_tps_per_mw=baseline,baseline='Maximum across all configurations; not NVIDIA baseline',
        x_not_normalized=True,interpolated_points=0))
    write(output/'selected-points.json',dict(
        pareto={str(a):[r['case'] for r in envelope[a]] for a in A_SIZES},
        best_per_concurrency={str(a):[r['case'] for r in best[a]] for a in A_SIZES},
        whole_batch={str(a):[r['case'] for r in reference[a]] for a in A_SIZES}))
    return envelope,best,reference


def finalize(args,completed,started,regression):
    rows=sorted(completed.values(),key=lambda r:(r['a_gpus'],r['batch'],r['af_batch_size']))
    write(args.output/'summary.json',dict(status='complete',model='DeepSeek-V3 671B BF16',
        context=4096,a_sizes=A_SIZES,f_gpus=32,batches=BATCHES,af_caps=AF_CAPS,
        includes_whole_batch_control=True,steps=1,router='F',combine='A',rows=rows,
        unique_case_count=len(rows),calibration='mixed measured exact shapes and analytical missing shapes',
        limitations=['Synchronous single decode step; no prefill or serving CPU overhead.',
            'MLA cache/down-projection replication is retained across A TP.',
            'Multi-A kernel shapes are largely analytical; speedup is not a measured hardware speedup.',
            'Uniform synthetic routing, BF16, no MTP/speculative decoding.',
            '25 GB/s shared virtual fabric and 5 us submission assumptions; not measured multirack networking.',
            'Fixed 250 W/card assumption; no CPU/network power. Not a reproduction of NVIDIA 2T Rubin/LP30.']))
    envelope,best,reference=plot(args.output,rows)
    lines=['# Full V3 A/F topology sweep','',f'{len(rows)} distinct simulated cases; every case has a validated profiling file.',
           '', '4K context; 1/2/4/8 A tensor-parallel ranks + 32 F ranks; real 80 GiB capacity checks.',
           'BF16; Router F / Combine A; AF cap 1/2/4/8/16/32 plus whole-batch controls.',
           'One synchronous decode step per case; mixed measurement and theory. GPU assumed power 250 W/card.',
           '', '| Topology | Max TPS/user | Max total TPS | Best pipeline gain at same concurrency |',
           '|---|---:|---:|---:|']
    for a in A_SIZES:
        subset=[r for r in rows if r['a_gpus']==a]
        gain=max(x['total_tokens_per_s']/y['total_tokens_per_s'] for x,y in zip(best[a],reference[a]))
        lines.append(f"| {a}A32F | {max(r['tokens_per_s_per_user'] for r in subset):.3f} | {max(r['total_tokens_per_s'] for r in subset):.3f} | {gain:.3f}x |")
    lines += ['', 'Multi-A shapes are largely theoretical. Treat differences as model predictions pending measurement, not isolated measured TP speedups.',
              'All plots share the same energy reference; absolute TPS/MW is also exported. NVIDIA absolute baseline is not used.',
              '', 'cases/*/traces contains every simulated profile; case summary/config/provenance/backlog and operator tables are retained.',
              'selected-points.json identifies the exact cases used for plotted frontiers and whole-batch comparisons.']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(args.output/'validation.json',dict(status='passed',cases=len(rows),
        history_regression=regression,each_case_validation='cases/*/validation.json'))
    write(args.output/'status.json',dict(status='complete',completed=len(rows),total=len(rows),
        elapsed_seconds=time.time()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--measurement-root',type=Path,required=True)
    parser.add_argument('--reference-summary',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    args.output=args.output.resolve(); args.measurement_root=args.measurement_root.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.workers < 1: parser.error('Workers must be positive')
    started=time.time(); completed={}; all_cases=cases(); regression=[]
    smoke=[(1,1,1),(1,8,8),(1,32,8),(2,4,2),(4,4,2),(8,4,2)]
    write(args.output/'experiment.json',dict(a_sizes=A_SIZES,f_gpus=32,batches=BATCHES,af_caps=AF_CAPS,
        unique_cases=[case_name(c) for c in all_cases],workers=args.workers,smoke_cases=smoke,
        physical_capacity_bytes=80*2**30,context=4096,steps=1,assumed_power_w_per_gpu=250))
    def collect(jobs,phase,workers):
        write(args.output/'status.json',dict(status='running',phase=phase,completed=len(completed),total=len(all_cases)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(run_case,args,case):case for case in jobs}
            for future in concurrent.futures.as_completed(pending):
                case=pending[future]; row=future.result(); completed[case]=row
                write(args.output/'status.json',dict(status='running',phase=phase,completed=len(completed),
                    total=len(all_cases),last_completed=case_name(case),elapsed_seconds=time.time()-started))
                write(args.output/'results.partial.json',list(completed.values()))
                print(f'{len(completed)}/{len(all_cases)} {case_name(case)}: '
                      f"{row['tokens_per_s_per_user']:.3f} TPS/user",flush=True)
    try:
        if shutil.disk_usage(args.output).free < 20*2**30:
            raise RuntimeError('Less than 20 GiB free; insufficient headroom for the full trace sweep')
        collect(smoke,'smoke',min(4,args.workers))
        old=json.loads(args.reference_summary.read_text())['rows']
        for b,af in ((1,1),(8,8),(32,8)):
            reference=next(r for r in old if r['batch']==b)
            difference=abs(completed[(1,b,af)]['avg_decode_step_ms']-reference['avg_decode_step_ms'])
            if difference > 1e-6: raise AssertionError(f'1A baseline changed at B={b}: {difference} ms')
            regression.append(dict(batch=b,difference_ms=difference))
        write(args.output/'smoke-validation.json',dict(status='passed',history_regression=regression))
        collect([c for c in all_cases if c not in completed],'full-sweep',args.workers)
        finalize(args,completed,started,regression)
        print('All topology/AF cases and profiles validated.',flush=True)
    except BaseException as error:
        write(args.output/'status.json',dict(status='failed',error=repr(error),completed=len(completed),
            total=len(all_cases),elapsed_seconds=time.time()-started))
        raise


if __name__=='__main__':
    main()
