"""Independent eager BF16 operator measurements; one explicitly selected GPU.

Uses operator libraries, never starts a model server or a vLLM engine.
Each case is independently checked, timed without profiler, and then profiled.
"""
import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

GPU_UUID = "GPU-76ca30e6-17c9-9a38-cb06-57aebeaa6973"


def dump(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def gpu_snapshot():
    return subprocess.check_output(["nvidia-smi", "-i", GPU_UUID,
        "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw",
        "--format=csv"], text=True).strip()


def check_exclusive():
    raw = subprocess.check_output(["nvidia-smi", "-i", GPU_UUID,
        "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True)
    others = [line for line in raw.splitlines() if line.strip() and int(line.split(",")[0]) != os.getpid()]
    if others:
        raise RuntimeError("Target GPU now has another process; stop without touching it: " + str(others))


def setup():
    global torch, F, triton, tl, ops, rope_kernel
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    import torch.nn.functional as F
    import triton
    import triton.language as tl
    from vllm import _custom_ops as ops
    torch.set_num_threads(1)
    torch.manual_seed(20260916)
    torch.backends.cuda.matmul.allow_tf32 = False

    @triton.jit
    def rope_kernel(X, O, COS, SIN, WIDTH: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
        dim = idx % 64
        partner = idx + tl.where(dim < 32, 32, -32)
        angle = (idx // WIDTH)*32 + dim % 32
        x = tl.load(X+idx, idx < TOTAL, other=0).to(tl.float32)
        y = tl.load(X+partner, idx < TOTAL, other=0).to(tl.float32)
        c = tl.load(COS+angle, idx < TOTAL, other=0)
        s = tl.load(SIN+angle, idx < TOTAL, other=0)
        tl.store(O+idx, x*c + tl.where(dim < 32, -y, y)*s, idx < TOTAL)


def rand(*shape):
    return torch.empty(shape, device="cuda", dtype=torch.bfloat16).normal_(std=0.02)


def check_close(actual, reference, *, atol=0.003, rtol=0.03):
    torch.testing.assert_close(actual.float(), reference.float(), atol=atol, rtol=rtol)
    relative_l2 = ((actual.float()-reference.float()).norm() / reference.float().norm().clamp_min(1e-12)).item()
    if relative_l2 > rtol:
        raise AssertionError(f"relative L2 error {relative_l2} exceeds {rtol}")
    return dict(passed=True, max_abs_error=(actual.float()-reference.float()).abs().max().item(),
                relative_l2_error=relative_l2,atol=atol, rtol=rtol, reference="independent FP32 arithmetic, TF32 disabled")


def make_case(case):
    table, key = case["table"], case["key"]
    if table == "gemm":
        x, w = rand(key["m"], key["k"]), rand(key["n"], key["k"])
        out = torch.empty((key["m"], key["n"]), device="cuda", dtype=torch.bfloat16)
        function = lambda: torch.mm(x, w.t(), out=out)
        validation = check_close(function(), x.float() @ w.float().t())
        return function, "torch.mm BF16/cuBLAS, preallocated output", validation
    if table == "batched_gemm":
        g, m, n, k = (key[s] for s in ("groups", "m", "n", "k"))
        x, w = rand(g,m,k), rand(g,k,n)
        out = torch.empty((g,m,n), device="cuda", dtype=torch.bfloat16)
        function = lambda: torch.bmm(x, w, out=out)
        # Independent head calls outside timing check all heads against FP32.
        ref = torch.stack([x[h].float() @ w[h].float() for h in range(g)])
        return function, "torch.bmm independent head matrices; one batched call", check_close(function(), ref)
    if table == "mla_attention":
        return make_mla(case)
    if table in ("moe", "moe_group"):
        return make_experts(case)
    if table != "elementwise":
        raise ValueError(f"unsupported table {table}")
    t, h, op = key["num_tokens"], key["hidden_size"], key["op_name"]
    x, y = rand(t,h), rand(t,h)
    out = torch.empty_like(x)
    if op == "rmsnorm":
        weight = torch.ones(h, device="cuda", dtype=torch.bfloat16)
        def function():
            ops.rms_norm(out, x, weight, 1e-6)
            return out
        ref = x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
        return function, "native RMSNorm; epsilon=1e-6", check_close(function(), ref, atol=.02)
    if op == "rope":
        assert h == (128+1)*64, "RoPE benchmark scoped to V3 H=128 and shared K rope=64"
        angle = (torch.arange(t,device="cuda")[:,None]+1) * (10000.0**(-torch.arange(0,64,2,device="cuda")/64))[None,:]
        cos, sin = angle.cos().contiguous(), angle.sin().contiguous()
        def function():
            rope_kernel[(triton.cdiv(t*h,256),)](x,out,cos,sin,h,t*h,256)
            return out
        a, b = x.float().reshape(t,129,64).chunk(2,dim=-1)
        ref = torch.cat((a*cos[:,None]-b*sin[:,None],b*cos[:,None]+a*sin[:,None]),-1).reshape(t,h)
        return function, "Triton out-of-place RoPE Q128+shared K1, dim64; precomputed cos/sin", check_close(function(),ref)
    if op in ("residual_add", "shared_sum"):
        return lambda: torch.add(x,y,out=out), "torch.add", check_close(torch.add(x,y),x.float()+y.float())
    if op == "routing_weight_multiply":
        weight = torch.rand((t,1),device="cuda",dtype=torch.bfloat16)
        return lambda: torch.mul(x,weight,out=out), "torch.mul route weights; assignments already expanded", check_close(x*weight,x.float()*weight.float())
    if op == "routed_reduce":
        # Provider encodes B*(top_k-1) scalar-vector additions, not B input rows.
        assert t % 7 == 0, "V3 top-8 reduction key must contain 7*B additions"
        partials=rand(t//7,8,h)
        out=torch.empty((t//7,h),device="cuda",dtype=torch.bfloat16)
        function=lambda: torch.sum(partials,dim=1,out=out)
        return function, "torch.sum top-8; key num_tokens=7*output_tokens", check_close(function(),partials.float().sum(1))
    raise ValueError(op)


def make_mla(case):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    k=case["key"]
    assert k["phase"] == "decode" and k["execution_mode"] == "absorbed"
    b,h,r,dr=(k[n] for n in ("batch_size","num_heads","kv_lora_rank","rope_head_dim"))
    contexts=[int(n) for n in k["context_lengths"].split(",")]
    assert len(set(contexts)) == 1
    s=contexts[0]
    page=16
    assert s % page == 0
    q,qp=rand(b,h,r),rand(b,h,dr)
    c,kp=rand(b*s//page,page,r),rand(b*s//page,page,dr)
    new_c,new_k=rand(b,r),rand(b,dr)
    slots=torch.arange(1,b+1,device="cuda",dtype=torch.int64)*s-1
    cf,kf=c.view(-1,r),kp.view(-1,dr)
    out=torch.empty((b,h,r),device="cuda",dtype=torch.bfloat16)
    wrapper=BatchMLAPagedAttentionWrapper(torch.empty(128*1024*1024,device="cuda",dtype=torch.uint8),backend="fa2")
    ind=torch.arange(b+1,device="cuda",dtype=torch.int32)
    scale=(k["nope_head_dim"]+dr)**-.5
    wrapper.plan(ind,ind*(s//page),torch.arange(b*s//page,device="cuda",dtype=torch.int32),
        torch.full((b,),s,device="cuda",dtype=torch.int32),h,r,dr,page,False,scale,
        torch.bfloat16,torch.bfloat16)
    def function():
        # Cache update belongs to core contract; input stays fixed across repeats.
        cf.index_copy_(0,slots,new_c)
        kf.index_copy_(0,slots,new_k)
        return wrapper.run(q,qp,c,kp,out=out)
    actual=function()
    refs=[]
    # Request-wise FP32 reference is used only for validation, never for timing.
    for i in range(b):
        ci,ki=cf[i*s:(i+1)*s].float(),kf[i*s:(i+1)*s].float()
        logits=(q[i].float()@ci.t()+qp[i].float()@ki.t())*scale
        refs.append(torch.softmax(logits,-1)@ci)
    validation=check_close(actual,torch.stack(refs),atol=0.0002,rtol=.03)
    validation.update(all_requests_checked=True,cache_write_checked=bool(torch.equal(cf[slots],new_c) and torch.equal(kf[slots],new_k)))
    assert validation["cache_write_checked"]
    return function, "FlashInfer 0.6.14 MLA FA2 absorbed BF16 + 2 index_copy cache updates; split latent/RoPE tensors; page16; planning excluded",validation


def make_experts(case):
    import triton.language as tl
    moe=importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    key=case["key"]
    counts=list(map(int,key["expert_token_counts"].split(","))) if case["table"]=="moe_group" else [key["num_tokens"]]
    tokens,hidden,inter=key["num_tokens"],key["hidden_size"],key["inter_size"]//key["tp_size"]
    experts=len(counts)
    x,w1,w2=rand(tokens,hidden),rand(experts,2*inter,hidden),rand(experts,hidden,inter)
    ids=torch.tensor([e for e,count in enumerate(counts) for _ in range(count)],device="cuda",dtype=torch.int32).view(tokens,1)
    weights=torch.ones((tokens,1),device="cuda",dtype=torch.float32)
    config=moe.try_get_optimal_moe_config(w1.size(),w2.size(),1,None,tokens,block_shape=None)
    up=torch.empty((tokens,1,2*inter),device="cuda",dtype=torch.bfloat16)
    activated=torch.empty((tokens,inter),device="cuda",dtype=torch.bfloat16)
    down=torch.empty((tokens,1,hidden),device="cuda",dtype=torch.bfloat16)
    kwargs=dict(compute_type=tl.bfloat16,use_fp8_w8a8=False,use_int8_w8a8=False,
                use_int8_w8a16=False,use_int4_w4a16=False,per_channel_quant=False)
    def function():
        sorted_ids,expert_ids,padded=moe._prepare_expert_assignment(ids,config,tokens,1,experts,None,
            use_int8_w8a16=False,use_int4_w4a16=False,block_shape=None,ignore_invalid_experts=True)
        moe.dispatch_fused_moe_kernel(x,w1,up,None,None,None,weights,sorted_ids,expert_ids,padded,False,1,config,**kwargs)
        moe.apply_moe_activation(moe.MoEActivation.from_str("silu"),activated,up.view(tokens,2*inter))
        moe.dispatch_fused_moe_kernel(activated,w2,down,None,None,None,weights,sorted_ids,expert_ids,padded,False,1,config,**kwargs)
        return down.view(tokens,hidden)
    actual=function()
    cursor=0
    max_error=0.
    for e,count in enumerate(counts):
        gate,up_ref=F.linear(x[cursor:cursor+count].float(),w1[e].float()).chunk(2,dim=-1)
        ref=F.linear(F.silu(gate)*up_ref,w2[e].float())
        valid=check_close(actual[cursor:cursor+count],ref,atol=.03,rtol=.05)
        max_error=max(max_error,valid["max_abs_error"])
        cursor+=count
    return function, "native grouped expert assignment + gate/up + SiLU + down; unweighted partials, no moe_sum",dict(
        passed=True,max_abs_error=max_error,atol=.03,rtol=.05,all_experts_checked=experts,
        expert_token_counts=counts,reference="independent per-expert FP32 dense arithmetic",kernel_config=config)


def mathematical_mla_check():
    """Absorbed vs expanded math before BF16 kernel calibration."""
    b,h,s,r,dn,dr,dv=2,4,17,16,8,4,8
    c,q,qp,kp=rand(b,s,r).float(),rand(b,h,dn).float(),rand(b,h,dr).float(),rand(b,s,dr).float()
    wk,wv=rand(h,r,dn).float(),rand(h,r,dv).float()
    keys=torch.einsum('bsr,hrd->bhsd',c,wk)
    vals=torch.einsum('bsr,hrd->bhsd',c,wv)
    score=(torch.einsum('bhd,bhsd->bhs',q,keys)+torch.einsum('bhd,bsd->bhs',qp,kp))*(dn+dr)**-.5
    expanded=torch.einsum('bhs,bhsd->bhd',score.softmax(-1),vals)
    absorbed_q=torch.einsum('bhd,hrd->bhr',q,wk)
    latent_scores=(torch.einsum('bhr,bsr->bhs',absorbed_q,c)+torch.einsum('bhd,bsd->bhs',qp,kp))*(dn+dr)**-.5
    latent=torch.einsum('bhs,bsr->bhr',latent_scores.softmax(-1),c)
    restored=torch.einsum('bhr,hrd->bhd',latent,wv)
    return check_close(restored,expanded,atol=1e-6,rtol=1e-5)


def measure(function, window_seconds):
    for _ in range(20): function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    blocks=[]
    windows=[]
    for window in range(3):
        check_exclusive()
        begun=time.perf_counter()
        calls=0
        while calls<100 or time.perf_counter()-begun<window_seconds:
            n=100
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter_ns()
            start.record()
            for _ in range(n): function()
            end.record()
            end.synchronize()
            blocks.append(dict(window=window,calls=n,device_interval_us=start.elapsed_time(end)*1000/n,
                               eager_wall_us=(time.perf_counter_ns()-wall)/1000/n))
            calls+=n
        current=[v for v in blocks if v["window"]==window]
        windows.append(dict(calls=calls,wall_seconds=time.perf_counter()-begun,
            device_interval_us=statistics.mean(v["device_interval_us"] for v in current),
            eager_wall_us=statistics.mean(v["eager_wall_us"] for v in current)))
    vals=sorted(v["device_interval_us"] for v in blocks)
    return dict(device_interval_us=statistics.median(v["device_interval_us"] for v in windows),
        eager_wall_us=statistics.median(v["eager_wall_us"] for v in windows),
        chunk_mean_p50_us=statistics.median(vals),chunk_mean_p95_us=vals[min(len(vals)-1,math.ceil(.95*len(vals))-1)],
        windows=windows,blocks=blocks,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),profiler_during_timing=False,cuda_graph=False)


def main():
    global GPU_UUID
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--gpu-uuid", default=GPU_UUID)
    p.add_argument("--window-seconds",type=float,default=2.)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    GPU_UUID=args.gpu_uuid
    if args.window_seconds <= 0:
        p.error("window-seconds must be positive")
    args.root.mkdir(parents=True,exist_ok=True)
    (args.root/'traces').mkdir(exist_ok=True)
    check_exclusive()
    # Prevent two copies of our own benchmark from racing for the same device.
    import fcntl
    lock=(args.root/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    setup()
    before=gpu_snapshot()
    cases=json.loads((args.root/'cases.json').read_text())
    if args.smoke:
        selected={}
        for c in cases:
            family=(c['table'],c['key'].get('op_name',''))
            if family not in selected: selected[family]=c
        cases=list(selected.values())
    resultfile=args.root/('smoke-measurements.json' if args.smoke else 'measurements.json')
    contractfile=args.root/('smoke-contract.json' if args.smoke else 'measurement-contract.json')
    input_sha256=hashlib.sha256((args.root/'cases.json').read_bytes()).hexdigest()
    if resultfile.exists():
        old=json.loads(contractfile.read_text())
        if old['gpu_uuid'] != GPU_UUID or old['input_sha256'] != input_sha256:
            raise RuntimeError('Refusing to mix different GPUs or input cases in existing results')
    rows=json.loads(resultfile.read_text()) if resultfile.exists() else []
    done={r['case_id'] for r in rows if r.get('status')=='ok'}
    physical_index=int(subprocess.check_output(['nvidia-smi','-i',GPU_UUID,'--query-gpu=index','--format=csv,noheader'],text=True).strip())
    contract=dict(gpu_uuid=GPU_UUID,physical_gpu_index=physical_index,torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),compute_capability=torch.cuda.get_device_capability(),
        seed=20260916,cuda_graph=False,profiler_during_timing=False,window_seconds=args.window_seconds,
        repeats=3,warmup=20,timing='eager CUDA-event intervals over 100 calls, including device gaps/host starvation; host wall separately reported',
        statistics='3 window means; P50/P95 refer to 100-call chunk means, not per-request tail latency',
        allocations='weights/inputs/output buffers outside timing; expert assignment scratch allocation included',
        cache_regime='warm repeated weights; no artificial L2 flush',
        mla_boundary='cache updates + absorbed core; excludes Q absorption/value expansion/projections and wrapper planning',
        expert_boundary='unweighted partials; excludes router/weighted combine',
        missing_scope='No V3 noaux_tc selection, no full engine CPU scheduling, no communication; no model checkpoint loaded',
        model_scope='V3 single-layer shapes, synthetic inputs; not model quality validation',
        input_sha256=input_sha256,
        gpu_before=before,mathematical_mla=mathematical_mla_check())
    dump(args.root/('smoke-contract.json' if args.smoke else 'measurement-contract.json'),contract)
    with torch.inference_mode():
        for case in cases:
            if case['case_id'] in done: continue
            check_exclusive()
            function=None
            try:
                function,implementation,validation=make_case(case)
                timing=measure(function,args.window_seconds)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                    with torch.profiler.record_function(case['case_id']+':'+case['table']):
                        for _ in range(3): function()
                        torch.cuda.synchronize()
                trace=args.root/'traces'/(("smoke-" if args.smoke else "")+case['case_id']+'.trace.json')
                prof.export_chrome_trace(str(trace))
                # Verify CUDA activities exist; CPU-only trace is not profiling evidence.
                events=json.loads(trace.read_text())['traceEvents']
                gpu_events=[e for e in events if e.get('cat')=='kernel']
                if not gpu_events: raise RuntimeError('Profiler produced no GPU kernel events')
                row=dict(case,status='ok',implementation=implementation,validation=validation,**timing,
                    trace=str(trace.relative_to(args.root)),profile_gpu_kernel_events=len(gpu_events),gpu_snapshot=gpu_snapshot())
            except Exception as exc:
                row=dict(case,status='failed',error=repr(exc),traceback=traceback.format_exc())
            rows=[r for r in rows if r['case_id']!=case['case_id']]+[row]
            dump(resultfile,rows)
            dump(args.root/'progress.json',dict(pid=os.getpid(),mode='smoke' if args.smoke else 'full',
                completed=sum(r['status']=='ok' for r in rows),failed=sum(r['status']!='ok' for r in rows),
                planned=len(cases),last_case=case['case_id'],last_status=row['status'],updated=time.strftime('%Y-%m-%d %H:%M:%S %z')))
            print(case['case_id'],case['table'],row['status'],round(row.get('device_interval_us',0),3),flush=True)
            if row['status']!='ok': print(row['error'],flush=True)
            function=None
            gc.collect()
            torch.cuda.empty_cache()
    contract['gpu_after']=gpu_snapshot()
    contract['completed']=sum(r['status']=='ok' for r in rows)
    contract['failed']=sum(r['status']!='ok' for r in rows)
    dump(args.root/('smoke-contract.json' if args.smoke else 'measurement-contract.json'),contract)
    if contract['failed']: raise SystemExit(1)


if __name__=='__main__':
    main()
