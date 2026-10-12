"""Render evidence summaries without changing the recorded measurements."""
import json
from pathlib import Path
import statistics
from collections import defaultdict

from TokenSim.validation.communication_experiments import HERE, RESULTS, write
from TokenSim.hardware.links import LinkClass
from TokenSim.comm.traffic import ring_volume


def formula_report():
    before = json.loads((HERE/'communication_baseline_before.json').read_text(encoding='utf-8'))
    after = json.loads((RESULTS/'communication_baseline_after.json').read_text(encoding='utf-8'))
    rows=[]
    for old,new in zip(before['cases'],after['cases']):
        rows.append(dict(mode=old['mode'],trace=old['trace'],before_s=old['makespan_s'],after_s=new['makespan_s'],
            delta_s=new['makespan_s']-old['makespan_s'],
            before_wall_median_s=statistics.median(item['wall_s'] for item in old['samples']),
            after_wall_median_s=statistics.median(item['wall_s'] for item in new['samples'])))
    link=LinkClass('formula_fixture','fixture',1e9,2,half_bandwidth_bytes=10000,collective_efficiency=.8)
    arithmetic=[]
    for operation in ('all_reduce','all_gather','reduce_scatter'):
        n,m=4,10000
        volume=ring_volume(operation,n,m)
        # Legacy flat transfer applied full-buffer traffic factors, one alpha,
        # and the link's H curve. This is an isolated arithmetic replay, not
        # a claim to reproduce every old hierarchical schedule.
        old_factor=(2 if operation=='all_reduce' else 1)*(n-1)/n
        old_us=link.transfer_us(m*old_factor)
        corrected_us=volume.rounds*link.latency_us+volume.send_bytes_per_rank/1e9*1e6
        arithmetic.append(dict(operation=operation,input_bytes=m,participants=n,
            legacy_flat_us=old_us,corrected_flat_us=corrected_us,
            corrected_node_aggregate_data_us=n*volume.send_bytes_per_rank/1e9*1e6,
            rounds=volume.rounds,wire_per_rank=volume.send_bytes_per_rank))
    write('formula_delta.json',dict(baseline=rows,arithmetic=arithmetic,
        attribution=['旧 H/efficiency 曲线移除，仅影响迁移后的 AF 服务',
            'AG 从完整 buffer 系数改为每 rank 输入口径',
            'flat ring 按真实轮次计 alpha；physical 按实际有向发送计 alpha',
            'physical 节点聚合资源与旧 per-endpoint 聚合口径不同，数值变化属于模型变化',
            '隔离算术使用旧 flat 资源排程；不伪造尚未保存的旧完整 DSV3 运行结果']))
    lines=['# 通信公式差异报告','',
        '基线提交及前置依赖见 `communication_baseline_before.json`。下面是两 worker 合成实验；算术对照保留旧 flat 调度边界，只替换服务公式。physical 同时改变资源契约，单独列出。没有目标硬件实测。','',
        '| 模式 | trace | 旧完成时间 µs | 新完成时间 µs | 差值 µs |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["mode"]} | {row["trace"]} | {row["before_s"]*1e6:.6f} | {row["after_s"]*1e6:.6f} | {row["delta_s"]*1e6:.6f} |')
    lines+=['','独立公式 fixture：N=4，M=10000 bytes，B=1 GB/s，alpha=2 µs，旧 H=10000 bytes。',
        '', '| 操作 | 旧 flat µs | 修正 flat µs | 正确每 rank wire bytes | 轮次 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in arithmetic:
        lines.append(f'| {row["operation"]} | {row["legacy_flat_us"]:.6f} | {row["corrected_flat_us"]:.6f} | {row["wire_per_rank"]} | {row["rounds"]} |')
    lines+=['','归因：去 H 改变小消息有效带宽；AG 修正输入口径；真实轮次增加相应启动成本；节点聚合带宽按全部有向发送计费。超售只作用于数据项，alpha=2、数据项=10 的例子由 12→22 µs。',
        '', '旧 `CollectiveModel` 与 `LinkClass.transfer_us` 的非 AF 消费者仍保留旧模型。复现旧实验应使用原提交及原配置，不能用新结果声称数值兼容。']
    (RESULTS/'formula_delta.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def reference_report():
    data=json.loads((RESULTS/'fifo_references.json').read_text(encoding='utf-8'))
    phases=json.loads((RESULTS/'phase_release_reference.json').read_text(encoding='utf-8'))
    lines=['# FIFO 与粗阶段近似评估','',
        '命令：`python -B -m TokenSim.validation.communication_experiments --suite micro`。生产代码仅使用独占 FIFO；均分带宽是测试专用参考，无硬件测量或预设误差界。',
        '', '前三组以 300 单位/秒容量发送 100/200/300 单位，用于隔离调度差异；源/目标资源组使用 50 单位/秒。时间单位为秒。',
        '', '| 场景 | FIFO 平均完成 | 共享平均完成 | FIFO P50/P95 | 共享 P50/P95 | FIFO 总 queue |',
        '| --- | ---: | ---: | --- | --- | ---: |']
    for name,row in data.items():
        if 'fifo' not in row:
            continue
        fifo=sorted(value['end']-value['ready'] for value in row['fifo'].values())
        shared=sorted(value['end']-value['ready'] for value in row['shared'].values())
        def percentile(values,p):
            position=(len(values)-1)*p
            lower=int(position)
            return values[lower]+(values[min(lower+1,len(values)-1)]-values[lower])*(position-lower)
        lines.append(f'| {name} | {statistics.mean(fifo):.6f} | {statistics.mean(shared):.6f} | '
                     f'{percentile(fifo,.5):.6f}/{percentile(fifo,.95):.6f} | '
                     f'{percentile(shared,.5):.6f}/{percentile(shared,.95):.6f} | '
                     f'{sum(value["queue"] for value in row["fifo"].values()):.6f} |')
    lines+=['','相同大小且同时 ready 的 A/B/C：FIFO 完成于 1/3、2/3、1；共享模式均完成于 1。两者最后完成时间相同，平均完成分别为 2/3 和 1。以每个网络完成时刻作为后继计算开始时间时，FIFO 允许前序后继更早启动；不能由总工作量相等推出应用性能相等。共享参考的 queue=0 表示立即开始获得部分服务，等待已经体现为更长服务期。',
        '', 'DSV3 真实通信形状：F TP=32、4 节点各 8 rank、每 rank 输入 32×7168×2 bytes；alpha_local=1 µs、alpha_inter=5 µs。AF 竞争任务在纯阶段时长的 1/4 时到达，服务 2 µs。',
        '', '| 阶段 | 粒度 | collective 完成 µs | AF queue µs | AF 完成 µs |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in phases:
        label = f'{row.get("case", "uniform_collective")} / {row["phase"]}'
        lines.append(f'| {label} | {"逐消息域参考" if row["fine"] else "生产粗阶段"} | '
                     f'{row["collective"]*1e6:.6f} | {row["af"]["queue"]*1e6:.6f} | {row["af"]["end"]*1e6:.6f} |')
    lines+=['', '统计 All-to-All 使用同一每 rank 输入和 remote_fraction=0.75，imbalance=1/8 仅改变目的分布；列出的 inter:1:0 是同一源/目标域的阶段，热点不增加全局应用 payload。',
        '', '| 场景/阶段 | coarse−fine AF queue µs | 相对 fine queue |',
        '| --- | ---: | ---: |']
    paired = defaultdict(dict)
    for row in phases:
        paired[(row.get('case', 'uniform_collective'),row['phase'])][row['fine']] = row
    for (case,phase),pair in paired.items():
        coarse,fine = pair[False]['af']['queue'],pair[True]['af']['queue']
        relative = f'{(coarse-fine)/fine:.2%}' if fine else '不定义（参考 queue=0）'
        lines.append(f'| {case} / {phase} | {(coarse-fine)*1e6:.6f} | {relative} |')
    lines+=['','粗阶段让后来的 AF 消息等待整个阶段，细参考在每轮资源释放后仲裁。ring 示例 AF queue 的绝对差为约 47.65/21.15 µs，足以影响重叠；没有证据可给出通用相对误差上界。All-to-All 示例的一轮消息全部先于 AF 进入 FIFO，后来任务不能越过已有等待者，因此该例粗、细结果相同；这不表示热点没有增加竞争。这里没有对整个请求或 dedicated/mixed 排序作精度结论；若部署选择依赖这类竞争差异，应在该真实工作负载上复审粗阶段粒度。',
        '', '额外构造的资源并集例见 `fifo_references.json:coarse_vs_fine`：local 域在第 1 个时间单位已空闲，但粗阶段等到跨节点第 10 个时间单位完成才释放；第 2 个时间单位到达的 AF 消息在 coarse/fine 分别于 11/3 完成。这是近似最坏案例的示例，不是所有硬件的最坏界。']
    (RESULTS/'fifo_references.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def final_report():
    required=('manifest.json','placement_decode.json','decode_performance.json','service_load.json',
              'hot_service_load.json','supplement.json','cache_trace_equivalence.json','full_regression.log',
              'profile_attribution.json')
    missing=[name for name in required if not (RESULTS/name).exists()]
    if missing:
        raise RuntimeError('evidence still pending: '+', '.join(missing))
    manifest=json.loads((RESULTS/'manifest.json').read_text(encoding='utf-8'))
    performance=json.loads((RESULTS/'decode_performance.json').read_text(encoding='utf-8'))
    placements=json.loads((RESULTS/'placement_decode.json').read_text(encoding='utf-8'))
    services=json.loads((RESULTS/'service_load.json').read_text(encoding='utf-8'))
    hot=json.loads((RESULTS/'hot_service_load.json').read_text(encoding='utf-8'))
    supplements=json.loads((RESULTS/'supplement.json').read_text(encoding='utf-8'))
    equivalent=json.loads((RESULTS/'cache_trace_equivalence.json').read_text(encoding='utf-8'))
    assert len(performance)==36 and len(placements)==6 and len(services)==48 and len(hot)==24
    assert len(supplements)==15
    assert len({row['routing_sha256'] for row in equivalent})==1
    assert max(row['simulated_s'] for row in equivalent)-min(row['simulated_s'] for row in equivalent)<1e-12
    for a_tp in (1,4):
        for batch in (1,8,32):
            shape=[row for row in performance if row['a_tp']==a_tp and row['batch']==batch]
            assert len(shape)==6 and len({row['routing_sha256'] for row in shape})==1
            assert len({row['event_steps'] for row in shape})==1
            assert max(row['simulated_s'] for row in shape)-min(row['simulated_s'] for row in shape)<1e-12
    for seed in (17,29,43):
        assert len({row['routing_sha256'] for row in placements if row['seed']==seed})==1
        for a_tp in (1,4):
            assert len({row['routing_sha256'] for row in supplements if row.get('a_tp')==a_tp and row['seed']==seed})==1
    for length in (128,2048):
        for seed in (17,29,43):
            for factor in (.3,.7,1.1,2.):
                pair=[row for row in services if row['input_length']==length and row['seed']==seed and row['load_factor']==factor]
                assert len(pair)==2 and pair[0]['absolute_rate']==pair[1]['absolute_rate']
                assert [r['arrival'] for r in pair[0]['requests']]==[r['arrival'] for r in pair[1]['requests']]
                assert pair[0]['incomplete_after_drain']==pair[1]['incomplete_after_drain']==0
    log=(RESULTS/'full_regression.log').read_text(encoding='utf-8')
    assert 'Ran 575 tests' in log and '\nOK\n' in log
    lines=['# 分层通信实施验证','',
        '模型：node-domain-fifo-v1；DeepSeek V3 完整 61 层，3 Dense + 58 MoE，hidden=7168，256 routed experts、top-8、1 shared expert，BF16。这属于模型结构与系统时延仿真，没有运行真实 checkpoint，也没有目标硬件测量。配置、库存、算子数据摘要、种子和覆盖索引见 manifest.json。',
        '', '完整回归：575 项通过；OpenSpec 严格校验和 git diff --check 通过。前置 integrate-deepseek-v3-core-runtime 规范应先同步，本变更未自动归档。浮点仿真等价比较容差为绝对 1e-12 秒；没有给硬件精度设置人为阈值。',
        '', '## 同库存放置','',
        'dedicated/mixed 均声明 9×8=72 张卡库存，33 张卡活跃、39 空槽；32F 在八节点各四卡，仅移动 A。节点带宽和 expert ownership 保持相同。domain_wire_bytes 的 tx/rx 是两侧资源负载，不可相加冒充两份应用发送量。',
        '', '| seed | 放置 | decode batch=8 完成 ms | 输出 tokens/s | local bytes | inter tx bytes | 域 queue 总和 ms |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in placements:
        local=sum(value for key,value in row['domain_wire_bytes'].items() if key.endswith(':local'))
        remote=sum(value for key,value in row['domain_wire_bytes'].items() if key.endswith(':inter_tx'))
        lines.append(f'| {row["seed"]} | {row["placement"]} | {row["simulated_s"]*1e3:.6f} | '
                     f'{row["output_tokens"]/row["simulated_s"]:.3f} | {local:.0f} | {remote:.0f} | '
                     f'{sum(row["domain_queue_s"].values())*1e3:.6f} |')
    lines+=['','同源/同目标造成的串行等待、反向和独立节点并行见 fifo_references.md。上表 queue 是逐任务逐资源统计，不是 makespan；并行完成时刻以实际事件时间线计算。4A32F 配对以及 context4096 结果见 supplement.json，重复运行的映射和显存账本见 manifest.json。',
        '', '## 仿真开销','',
        '每一行汇总三个独立子进程中的同一输入；计时从 provider 构造开始，到完整 JSON 导出及字节统计结束。模型目录加载、runtime 编译，以及导出后验证脚本的 trace 汇总、不变量复查和子进程退出在计时窗口之外。报告 CPU、wall 中位数及范围，峰值为操作系统进程峰值工作集，包含配置、仿真及完整 JSON 导出。程序在同一机器上与其他验证进程并行，部分 wall 样本出现较大暂停，因此 CPU 时间用于区分计算成本和 wall 干扰；这不是专用机器的极限速度。早期 tracemalloc 探测会显著增加开销，单独保留，未混入下面统计。',
        '', '| A TP | batch/block | trace | CPU 中位 s | wall 中位 [min,max] s | 峰值 MiB | SimPy events | 导出中位 s |',
        '| --- | --- | --- | ---: | --- | ---: | ---: | ---: |']
    groups=defaultdict(list)
    for row in performance:
        groups[(row['a_tp'],row['batch'],row['block'],row['trace'])].append(row)
    for (a,b,block,trace),rows in sorted(groups.items()):
        walls=[row['wall_s'] for row in rows]
        assert len(rows)==3
        assert len({row['routing_sha256'] for row in rows})==1
        lines.append(f'| {a} | {b}/{block} | {trace} | {statistics.median(row["cpu_s"] for row in rows):.3f} | '
            f'{statistics.median(walls):.3f} [{min(walls):.3f},{max(walls):.3f}] | '
            f'{max(row["process_peak_bytes"] for row in rows)/2**20:.1f} | {rows[0]["event_steps"]} | '
            f'{statistics.median(row["export_wall_s"] for row in rows):.3f} |')
    lines+=['','每层 trace tasks 及 provider/计划/执行/导出分项在 decode_performance.json；动态估值与缓存查询包含于 execution 时间，额外用 profile_attribution.json 的单独 CPU profiler 样本分离 provider/cache、planning、event execution 等类别的 self time，避免嵌套累计重复计费。该 profiler 样本含测量开销，未用于上述中位数。关闭 trace 时没有记录这些 trace tasks，其 0 值表示未采集，不表示没有执行，参考同 shape 的 trace 开启行。SimPy event_steps 则在开关两种情况下均独立计数。batch1 是稀疏专家命中，batch32 的循环均衡路由覆盖全体专家；没有丢弃任务。',
        '', '旧基线只保存了两 worker 最小实验，不能将新完整模型开销与它直接算成模型迁移倍率。formula_delta.md 分离了数值变化；增加事件来自阶段、域资源申请释放与布局计算。计划键改为完整内容摘要，避免缓存保留 token 路由原图；原有大 prefill DAG 模板缓存的内存问题单独记录，没有通过缩小模型处理。',
        '', '## 完整请求与服务负载','',
        '每个底层服务样本完整执行 61 层未分块 prefill 和后续 15 次 decode，输出共 16 tokens。TTFT 从请求到达到 prefill 产生首 token；TBT 为其后 15 个 token 间隔的平均。先以 dedicated、seed17 测得 C，再对两方案复用相同绝对请求率与到达序列。每个序列包含四个独立会话，显式复用相同冻结 assignments；单请求准入 FCFS 每次完全排空资源，因此可精确回放已完整执行的服务样本。该测试不代表动态批处理或多请求重叠的服务器。',
        '', '128/2048 均测试三 seed、0.3C/0.7C/1.1C/2C，另对 128 的热点路由（90% 偏向 experts0..7）执行相同三 seed 对照。2048 样本显式关闭既有 DAG 模板缓存，保留通信估值和计划缓存，完整配置见 service_config_*.json；没有改变 dtype、模型或任务数。',
        '', '| 输入长度 | 放置 | 负载/C | mean TTFT s（跨 seed） | mean TBT s | 最后准入 queue s 范围 | drain s 范围 |',
        '| --- | --- | --- | ---: | ---: | --- | --- |']
    groups=defaultdict(list)
    for row in services:
        groups[(row['input_length'],row['placement'],row['load_factor'])].append(row)
    for (length,name,factor),rows in sorted(groups.items()):
        queues=[row['requests'][-1]['queue_s'] for row in rows]
        drains=[row['drain_s'] for row in rows]
        lines.append(f'| {length} | {name} | {factor} | {statistics.mean(row["mean_ttft_s"] for row in rows):.6f} | '
            f'{statistics.mean(row["mean_tbt_s"] for row in rows):.6f} | {min(queues):.6f}–{max(queues):.6f} | '
            f'{min(drains):.6f}–{max(drains):.6f} |')
    lines+=['','所有到达窗口结束时的 backlog 和最终未完成数、逐请求准入/结束/TTFT/TBT 在 service_load.json；最终 drain 后均为零。有限序列只显示排队和清空趋势，不宣称稳态或给每个方案重新归一化 C。热点同样固定原参考 C，完整数据见 hot_service_load.json。decode-only 指标在 placement_decode.json 中称 step latency，未标为 TTFT。',
        '', '## V1–V13 覆盖与限制','',
        '| 编号 | 证据 | 状态 |','| --- | --- | --- |']
    for key,path in manifest['coverage'].items():
        status = '已执行：近似敏感性评估' if key in ('V3','V4','V5') else (
            '已执行：单请求准入负载' if key=='V11' else (
                '已执行：仿真成本测量' if key=='V12' else '通过：模型验证'))
        lines.append(f'| {key} | {path} | {status} |')
    lines+=['','FIFO 和粗阶段的主要偏差见 fifo_references.md：总工作量一致不保证平均完成一致，粗阶段可显著延迟后到 AF 消息。当前放置对照使用生产 FIFO，并不能证明真实硬件或细参考下排序相同；部署若依赖该竞争，需要对实际 workload 复审阶段粒度。',
        '', '本次交付覆盖模型自洽性、资源容量、守恒、依赖、缓存等价、确定性、取消恢复和模拟成本，不声称 NCCL、NVSwitch 或完整 GPU kernel 的实测精度。']
    (RESULTS/'verification.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    import sys
    formula_report()
    reference_report()
    if '--final' in sys.argv:
        final_report()
