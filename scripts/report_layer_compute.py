"""Compare layer services after moving residual costs to native fused boundaries."""
import json
import math
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
native = json.loads((root/'outputs/afd-compute-gpu03-20261006/layer-summary.json').read_text())
original = json.loads((root/'outputs/afd-compute-gpu03-20261006/compute-summary.json').read_text())
bundle = []
table = ['| Batch | Layer | A 模拟 / 实测 | F 模拟 / 实测 | 合计模拟 / 实测 | 实测/模拟 |',
         '| --- | --- | ---: | ---: | ---: | ---: |']
outside_table = ['| Batch | 层外模拟 | 层外实测 | 差额 |', '| --- | ---: | ---: | ---: |']
details = []
labels = dict(input_norm_previous_residual='输入norm＋上一层residual', qkv_projection='QKV投影',
              q_norm='Q norm', k_norm='K norm', rope='RoPE', attention_and_kv='Attention core＋KV写入',
              output_projection='输出投影', post_norm_attention_residual='Attention residual＋post norm',
              layout_copy='层内布局拷贝', router='Router', expert_combine='专家＋路由布局＋combine')
for b in (1, 4, 8):
    n = next(r for r in native if r['batch'] == b)
    old = next(r for r in original if r['batch'] == b)
    for role in ('attention', 'ffn'):
        assert not n[role]['unclassified']
        assert math.isclose(n[role]['median_us']['total'], old[role]['median_us']['compute'], abs_tol=1e-6)
        m = n[role]['mean_us']
        assert math.isclose(m['total'], m['layer0']+m['layer1']+m.get('outside', 0), abs_tol=1e-6)
    events = json.loads((root/f'outputs/afd-compute-baseline-20261006/tokensim-b{b}-decode-events.json').read_text())
    costs = defaultdict(float)
    raw = defaultdict(float)
    for e in events:
        if e['event_type'] != 'compute':
            continue
        layer = e['layer_id']
        raw[layer] += e['service_duration']*1e6/31
        for c in e['details']['components']:
            costs[layer, c['name']] += c['latency_us']/31
    # Sum names are read directly from the trace, so no new estimator is involved.
    per_layer = []
    for layer in (0, 1):
        components = {k[1]: v for k, v in costs.items() if k[0] == layer}
        # Operation-level values avoid depending on the routed_experts component label.
        by_op = defaultdict(float)
        for e in events:
            if e['event_type'] == 'compute' and e['layer_id'] == layer:
                by_op[e['operation']] += e['service_duration']*1e6/31
        prev_residual = 0.0 if layer == 0 else sum(e['service_duration']*1e6/31 for e in events
                      if e['event_type'] == 'compute' and e['layer_id'] == layer-1 and e['operation'] == 'residual')
        s = dict(input_norm_previous_residual=components['a_rmsnorm']+prev_residual,
                 qkv_projection=components['qkv_projection'], q_norm=components['q_norm'],
                 k_norm=components['k_norm'], rope=components['rope'],
                 attention_and_kv=components['attention_core'], output_projection=components['output_projection'],
                 post_norm_attention_residual=components['a_residual_add']+by_op['post_attention_norm'],
                 layout_copy=0.0, router=by_op['router_topk'],
                 expert_combine=by_op['routed_expert']+by_op['f_weighted_combine'])
        nm = n['attention']['mean_us']
        fm = n['ffn']['mean_us']
        measured = {k: nm.get(f'layer{layer}/{k}', 0) for k in s}
        measured['attention_and_kv'] = nm[f'layer{layer}/attention_core']+nm[f'layer{layer}/kv_cache_write']
        measured['router'] = fm[f'layer{layer}/router']
        measured['expert_combine'] = fm[f'layer{layer}/expert_and_routing_layout']+fm[f'layer{layer}/combine_sum']
        sa = sum(v for k,v in s.items() if k not in ('router','expert_combine'))
        sf = s['router']+s['expert_combine']
        na, nf = nm[f'layer{layer}'], fm[f'layer{layer}']
        assert math.isclose(sum(measured.values()), na+nf, abs_tol=1e-6)
        per_layer.append(dict(layer=layer, simulated_a_us=sa, simulated_f_us=sf, native_a_us=na,
                              native_f_us=nf, simulated_components=s, native_components=measured,
                              raw_logical_layer_simulated_us=raw[layer]))
        table.append(f'| {b} | {layer} | {sa:.2f} / {na:.2f} | {sf:.2f} / {nf:.2f} | {sa+sf:.2f} / {na+nf:.2f} | {(na+nf)/(sa+sf):.2f}× |')
        details.extend([f'### Batch {b}，Layer {layer}', '', '| 部分 | 模拟 µs | 实测 µs |', '| --- | ---: | ---: |'])
        details.extend(f'| {labels[k]} | {v:.3f} | {measured[k]:.3f} |' for k,v in s.items())
        details.append('')
    last_residual = sum(e['service_duration']*1e6/31 for e in events if e['event_type'] == 'compute'
                        and e['layer_id'] == 1 and e['operation'] == 'residual')
    sim_outside = raw[-1]+last_residual
    native_outside = n['attention']['mean_us']['outside']
    assert math.isclose(sum(raw.values()), sim_outside+sum(r['simulated_a_us']+r['simulated_f_us'] for r in per_layer), abs_tol=1e-6)
    outside_table.append(f'| {b} | {sim_outside:.2f} | {native_outside:.2f} | {native_outside-sim_outside:.2f} |')
    bundle.append(dict(batch=b, layers=per_layer, simulated_outside_us=sim_outside, native_outside_us=native_outside,
                       native_evidence=n))

out = root/'docs/plans/afd-layer-compute-alignment-20261006'
out.with_suffix('.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
intro = '''# 逐layer计算对照：TokenSim / vLLM+AFD

复用2026-10-06在PKUa100 GPU0/3采集的真实AFD trace，没有重新启动GPU测试。模型为此前的小型Qwen3 MoE，共2层，代码编号Layer 0和Layer 1；不能外推为DSV3逐层实测。

每个batch统计31个decode步（context 129–159），不含prefill、通信、CPU耗时和等待。本报告统一用每步kernel时长之和的**均值**，使层与层外成本可加；此前总表使用原生中位数，数值略有区别。A和F相加仅表示计算工作量，不是重叠后的实际层延迟。

## 逐层汇总

单位µs/该层/一次decode步。

'''
boundary = '''

为对应原生执行边界，模拟器原有Layer 0的MoE后residual费用移入Layer 1输入norm组，Layer 1的MoE后residual费用移入层外final norm组。这只是报告重分组，模拟器参数、事件和总成本均未修改。原生融合kernel不做任意拆分：Attention residual和post norm一起比较，MoE residual和下一个norm一起比较。未经重分组的模拟层总量保存在同名JSON的raw_logical_layer_simulated_us中。

## 层外GPU计算

'''
notes = '''

层外包括embedding、最终norm（融合最后一层residual）、logits/argmax，以及索引、拷贝、KV slot mapping等辅助GPU操作。它们每步执行，不能把这部分费用平均摊到每一层再拟合层算子。

## 已定位的差异

1. 两层计算耗时接近，没有某一层异常放大的迹象。逐层整体误差小于此前含层外成本的A侧倍率。
2. batch1每层A侧9个kernel；batch4/8每层12个，新增3个aten::copy_ kernel，每层合计约10–11µs。解析模型没有对应的独立布局拷贝项。这是batch增长差异的直接证据，不应只调大统一启动常数。
3. 原生每层Attention core约8.5–8.8µs，KV cache写入约2.9–3.1µs。模型generation_attention的kv_bytes已包含读写流量，但只用一个解析服务/启动项；不能说KV写入字节完全漏算。报告将原生core和cache写入合并，与模拟attention_core对比。
4. 模拟器把residual和norm分开计费，原生融合执行，部分norm组合反而高估；与此同时attention core、布局拷贝和部分投影低估。统一乘一个系数会掩盖不同方向的误差。
5. Router按原生gate至top-k结束的范围单独比较。专家、路由布局和combine作为整体比较，因为加权与下投影可能融合；不将moe_sum直接当作完整weighted_combine的实测值。
6. 层外实测约50–55µs，当前对应模型约15µs，差额约35–40µs。该差额属于本次GPU kernel口径，不等于CPU调度开销，更不属于网络传输。

## 切分与验证

原始AFD trace没有显式layer annotation。这里用已核对源码的顺序和形状还原：每个decode步A侧恰有5次hidden=256的norm调用（input0/post0/input1/post1/final），F侧恰有2次shape=[batch,256]的vllm::moe_forward。按调用顺序编号Layer 0/1。每个GPU kernel通过CUDA runtime/driver correlation匹配CPU提交，再归入层区间；无法匹配或结构不符会停止分析。

三个batch均识别31个decode步，无未分类层内kernel；层总和＋层外均值与总计算均值守恒，原生总计算中位数也与上一轮报告逐点一致。模拟重分组前后总成本一致。这是源码与形状支持的层序推断，不是原始trace自带的layer ID，也没有回放实际expert routing。

## 每层明细

布局拷贝模拟值0表示当前无独立费用条目，其他模型项是否间接吸收相关成本不在本次拟合范围。真实kernel时长受profiling影响，当前各batch只有一个AFD采样窗口；已有单卡三窗口对照只用于辅助验证总计算稳定性。

'''
ending = '''
## 证据和复现

- 同名JSON保存逐层模拟分项、重分组前层总量、原生均值/中位数/范围、kernel数量和copy shape计数。
- 远端原始trace：`/home/youwei/autodl-tmp/af-workspace/profiling/afd-compute-gpu03-20261006/c{1,4,8}/{attention,ffn}`。
- 远端执行：`envs/afd026/bin/python profiling/summarize_layer_compute.py profiling/afd-compute-gpu03-20261006`，只生成layer-summary.json，不使用GPU。
- 下载小摘要后，本地运行：`python scripts/report_layer_compute.py`。
- 模型位置：`TokenSim/af/operator_provider.py::_attention_components`、`TokenSim/operator_data/workload.py::generation_attention_work`、`TokenSim/operator_data/analytical/gpu.py::GPURooflineModel.estimate`。

本轮只增加诊断脚本和报告，没有修改估值公式、校准参数或提交代码。
'''
out.with_suffix('.md').write_text(intro+'\n'.join(table)+boundary+'\n'.join(outside_table)+notes+'\n'.join(details)+ending, encoding='utf-8')
print('\n'.join(table+['']+outside_table))
