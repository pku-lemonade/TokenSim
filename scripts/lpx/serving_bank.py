"""Admitted LP30 stage and boundary programs for complete request replay."""

from bisect import bisect_left
from pathlib import Path

from TokenSim.af.runtime.admission import ResidentKV
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.context_envelopes import reserve_contexts
from scripts.lpx.io_pool import pool_memory, transformer_component, transformer_runtime
from scripts.lpx.pool_ops import embedding, logits
from scripts.lpx.profile_stages import split_stages
from scripts.lpx.program import LP30Program
from scripts.lpx.replay_profiles import cached_profile, save_profile


class NativeServingBank:
    def __init__(
        self,
        config,
        workload,
        *,
        tp,
        kvp,
        pp,
        pool_chips,
        cache,
        bands=4,
        global_kv_layout="separate",
    ):
        integer(bands, "prefill context bands", minimum=1)
        for value, name in (
            (tp, "TP"),
            (kvp, "DCP"),
            (pp, "PP"),
            (pool_chips, "head chips"),
        ):
            integer(value, name, minimum=1)
        if workload.inputs + workload.outputs > 102000:
            raise ConfigurationError("serving workload exceeds the 102K reservation")
        self.workload = workload
        self.tp = tp
        self.pp = pp
        self.kvp = kvp
        self.pool_chips = pool_chips
        self.chips = tp * pp + pool_chips
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.hardware, self.model, self.runtime, self.admission = transformer_runtime(
            config,
            tp=tp,
            kvp=kvp,
            pp=pp,
            queries=min(workload.chunk, workload.inputs),
            logits_rows=1,
            resident=workload.clients,
            global_kv_layout=global_kv_layout,
        )
        self.pool_memory = pool_memory(
            self.model, pool_chips, min(workload.chunk, workload.inputs), logits_rows=1
        )
        self.heads = tuple(range(tp * pp, self.chips))
        self.frames = {}
        self.simple = {}
        self.receipts = []
        self.memory = ResidentKV(self.runtime.plan)
        self.owner = next(iter(self.runtime.plan.request_owners))
        self.contexts = {}
        for offset in range(0, workload.inputs, workload.chunk):
            width = min(workload.chunk, workload.inputs - offset)
            context = offset + width
            ceiling = min(
                workload.inputs,
                -(-(context * bands) // workload.inputs) * workload.inputs // bands,
            )
            # Integer bands may end one position below the requested context.
            ceiling = max(context, ceiling)
            self.contexts.setdefault((width, ceiling), []).append(context)
        if workload.outputs > 1:
            ceiling = workload.inputs + workload.outputs - 1
            self.contexts.setdefault((1, ceiling), []).extend(
                range(workload.inputs + 1, ceiling + 1)
            )
        self.ceilings = {
            q: sorted(c for width, c in self.contexts if width == q)
            for q, _ in self.contexts
        }

    def frame_key(self, task):
        q = task.queries
        if task.phase == "decode":
            return q, self.workload.inputs + self.workload.outputs - 1
        return q, self.ceilings[q][bisect_left(self.ceilings[q], task.context)]

    def key(self, kind, task, stage):
        if kind in {"body", "handoff"}:
            return kind, *self.frame_key(task), stage
        if kind in {"head", "head_copy"}:
            return kind
        return kind, task.queries, stage

    def frame(self, task):
        key = self.frame_key(task)
        if key not in self.frames:
            queries, context = key
            profile = cached_profile(
                self.runtime,
                "mfu-search",
                self.cache / "native",
                batch=1,
                queries=queries,
                phase="decode",
                emit_logits=False,
                record_node_ids=True,
                record_attention_keys=True,
                context=context,
                contexts=(context,),
                maximum_contexts=(102000,),
                dp_rank=0,
                seed=42,
                sample=0,
                streaming=dict(
                    overlap=False, early_output=False, collective_launch_us=0
                ),
            )
            profile = transformer_component(
                profile, tp=self.tp, cache=self.cache / "native"
            )
            reserved = reserve_contexts(
                profile, self.runtime, self.contexts[key], queries
            )
            reserved = save_profile(
                self.cache / f"envelope-q{queries}-c{context}.json.gz",
                reserved,
                dict(
                    parent=profile.metadata["input_sha256"],
                    contexts=self.contexts[key],
                    kind="serving-context-envelope-v1",
                ),
            )
            self.frames[key] = split_stages(reserved, tp=self.tp, pp=self.pp)
            self.receipts.append(
                dict(
                    queries=queries,
                    context_ceiling=context,
                    covered_contexts=len(self.contexts[key]),
                    input_sha256=reserved.metadata["input_sha256"],
                    node_body_sha256=reserved.metadata["node_body_sha256"],
                    source_sha256=reserved.metadata["source_sha256"],
                    context_reservations=reserved.metadata["context_reservations"],
                    stage_parts={
                        f"{s}:{p}": part.metadata["modeled_duration_s"]
                        for (s, p), part in self.frames[key].items()
                    },
                )
            )
        return self.frames[key]

    def profile(self, kind, task, stage):
        if kind in {"body", "handoff"}:
            return self.frame(task)[(stage, int(kind == "handoff"))]
        key = self.key(kind, task, stage)
        if key in self.simple:
            return self.simple[key]
        p = LP30Program(self.hardware, self.chips)
        rows = task.queries
        h = self.model.hidden_size
        if kind == "input":
            launched = p.append((), ("mfu-search/host:0",), duration=4e-6)
            ready = embedding(p, (launched,), self.heads, self.model, rows)
            p.transfer(
                (ready,),
                [
                    (self.heads[i % self.pool_chips], i, rows * h * 2)
                    for i in range(self.tp)
                ],
            )
        elif kind == "entry":
            for chip in range(stage * self.tp, (stage + 1) * self.tp):
                p.compute(
                    (),
                    chip,
                    "elementwise",
                    dict(
                        op_name="data_reorder",
                        dtype="bf16",
                        num_tokens=rows,
                        hidden_size=h,
                    ),
                )
        elif kind == "head_copy":
            p.transfer(
                (),
                [
                    ((self.pp - 1) * self.tp + i % self.tp, chip, h * 2)
                    for i, chip in enumerate(self.heads)
                ],
            )
        elif kind == "head":
            logits(p, (), self.heads, self.model, 1)
        else:
            raise ConfigurationError("unknown serving program kind")
        self.simple[key] = p.profile()
        return self.simple[key]

    def request_id(self, client, cycle):
        return cycle * self.workload.clients + client

    def admit(self, client, cycle):
        self.memory.admit(
            self.request_id(client, cycle),
            self.owner.domain,
            self.owner.instance,
            maximum_context=self.workload.inputs + self.workload.outputs,
            current_context=0,
        )

    def commit(self, client, cycle, context):
        self.memory.commit(self.request_id(client, cycle), context)

    def release(self, client, cycle):
        self.memory.release(self.request_id(client, cycle))

    def validate_empty(self):
        if self.memory.leases or any(self.memory.reserved.values()):
            raise SimulationStateError("complete serving leaked resident KV")

    def describe(self):
        return dict(
            chips=self.chips,
            layout=dict(
                tp=self.tp, kvp=self.kvp, pp=self.pp, pool_chips=self.pool_chips
            ),
            transformer=self.admission,
            pool_memory=self.pool_memory,
            frames=self.receipts,
            precision=dict(
                weights=self.model.dtype,
                activations=self.model.activation_dtype,
                kv=self.model.kv_cache_dtype,
            ),
            scope="Native multiquery transformer graphs represent causal prefill chunks; the production aggregate-prefill route is not used. Whole-buffer collectives, explicit entry copies, one active and one incoming frame per stage, and terminal-only prompt logits are modeled. Context budgets cover every requested position analytically; hardware banks, compiler execution and model quality remain unverified.",
        )
