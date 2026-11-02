"""Finite complete-request replay with bounded pipeline activation buffers."""

from collections import Counter, deque
from dataclasses import dataclass

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.replay_engine import ReplayKernel
from scripts.lpx.serving_metrics import latency_metrics


@dataclass(frozen=True)
class ServingWorkload:
    inputs: int = 100000
    outputs: int = 2000
    chunk: int = 256
    clients: int = 1
    cycles: int = 1

    def __post_init__(self):
        for name in ("inputs", "outputs", "chunk", "clients", "cycles"):
            integer(getattr(self, name), f"serving {name}", minimum=1)


@dataclass(frozen=True)
class ForwardTask:
    client: int
    cycle: int
    offset: int
    queries: int
    phase: str
    logits: bool

    @property
    def context(self):
        return self.offset + self.queries


class ServingPipeline:
    """One active frame and one reserved incoming frame per stage.

    The incoming slot is released only after an explicit entry copy finishes.
    The outgoing slot is reserved before stage execution. The final stage holds
    its output until the logits pool completes the hidden-state transfer.
    """

    def __init__(self, bank, workload):
        self.bank, self.workload = bank, workload
        self.kernel = ReplayKernel()
        self.profiles = {}
        self.pending = {}
        self.now = 0.0
        self.pp = bank.pp
        self.reserved = [False] * self.pp
        self.arrived = [None] * self.pp
        self.active = [None] * self.pp
        self.contexts = [[0] * self.pp for _ in range(workload.clients)]
        self.io = None
        self.wait_head = None
        self.input = deque(self.prefill(c, 0, 0) for c in range(workload.clients))
        self.generated = [0] * workload.clients
        self.counters = Counter()
        self.tokens = []
        self.requests = []
        self.starts = {(c, 0): 0.0 for c in range(workload.clients)}
        self.first = {}
        self.max_pending = [0] * self.pp

    def prefill(self, client, cycle, offset):
        width = min(self.workload.chunk, self.workload.inputs - offset)
        return ForwardTask(
            client,
            cycle,
            offset,
            width,
            "prefill",
            offset + width == self.workload.inputs,
        )

    def submit(self, kind, task, stage=0):
        key = self.bank.key(kind, task, stage)
        if key not in self.profiles:
            self.profiles[key] = self.kernel.add(
                self.bank.profile(kind, task, stage).nodes
            )
        identifier = self.kernel.submit(self.profiles[key], self.now)
        self.pending[identifier] = (kind, task, stage)
        self.counters[kind] += 1

    def dispatch(self):
        # Output priority prevents token feedback from waiting behind all prompts.
        if self.io is None and self.wait_head is not None:
            task = self.wait_head
            self.wait_head = None
            self.io = task
            self.submit("head_copy", task, self.pp - 1)
        elif self.io is None and self.input and not self.reserved[0]:
            task = self.input.popleft()
            if task.offset == 0:
                self.starts.setdefault((task.client, task.cycle), self.now)
                self.bank.admit(task.client, task.cycle)
            self.reserved[0] = True
            self.io = task
            self.submit("input", task)
        for stage in reversed(range(self.pp)):
            if self.active[stage] is not None or self.arrived[stage] is None:
                continue
            if stage + 1 < self.pp and self.reserved[stage + 1]:
                continue
            task = self.arrived[stage]
            self.arrived[stage] = None
            if self.contexts[task.client][stage] != task.offset:
                raise SimulationStateError("stage KV contexts are out of order")
            if stage + 1 < self.pp:
                self.reserved[stage + 1] = True
            self.active[stage] = task
            self.submit("entry", task, stage)
        self.max_pending = [
            max(a, int(b)) for a, b in zip(self.max_pending, self.reserved)
        ]

    def finish(self, kind, task, stage):
        if kind == "input":
            self.io = None
            self.arrived[0] = task
            if task.phase == "prefill" and not task.logits:
                self.input.append(self.prefill(task.client, task.cycle, task.context))
        elif kind == "entry":
            self.reserved[stage] = False
            self.submit("body", task, stage)
        elif kind == "body":
            self.contexts[task.client][stage] = task.context
            if stage + 1 < self.pp:
                self.submit("handoff", task, stage)
            else:
                self.bank.commit(task.client, task.cycle, task.context)
                if task.phase == "prefill":
                    self.counters["prefill_tokens"] += task.queries
                else:
                    self.counters["decode_tokens"] += 1
                if task.logits:
                    if self.wait_head is not None:
                        raise SimulationStateError("logits staging overflow")
                    self.wait_head = task
                else:
                    self.active[stage] = None
        elif kind == "handoff":
            self.active[stage] = None
            if self.arrived[stage + 1] is not None:
                raise SimulationStateError("pipeline input overflow")
            self.arrived[stage + 1] = task
        elif kind == "head_copy":
            self.active[stage] = None
            self.submit("head", task)
        elif kind == "head":
            self.io = None
            client, cycle = task.client, task.cycle
            self.generated[client] += 1
            count = self.generated[client]
            self.tokens.append(
                dict(client=client, cycle=cycle, token=count, time=self.now)
            )
            if count == 1:
                self.first[(client, cycle)] = self.now
            if count == self.workload.outputs:
                self.requests.append(
                    dict(
                        client=client,
                        cycle=cycle,
                        start=self.starts[(client, cycle)],
                        first_output=self.first[(client, cycle)],
                        last_output=self.now,
                    )
                )
                self.bank.release(client, cycle)
                if cycle + 1 < self.workload.cycles:
                    self.generated[client] = 0
                    self.contexts[client] = [0] * self.pp
                    self.starts[(client, cycle + 1)] = self.now
                    self.input.append(self.prefill(client, cycle + 1, 0))
            else:
                self.input.append(
                    ForwardTask(
                        client,
                        cycle,
                        self.workload.inputs + count - 1,
                        1,
                        "decode",
                        True,
                    )
                )
        else:
            raise SimulationStateError("unknown serving completion kind")

    def run(self):
        try:
            self.dispatch()
            while self.pending:
                done = self.kernel.advance()
                if done is None:
                    raise SimulationStateError("serving replay stalled")
                self.now = done.end
                self.finish(*self.pending.pop(done.id))
                self.dispatch()
            w = self.workload
            requests = w.clients * w.cycles
            if (
                len(self.requests) != requests
                or self.input
                or self.io
                or self.wait_head
                or any(self.reserved)
                or any(t is not None for t in self.active)
                or any(t is not None for t in self.arrived)
                or self.counters["prefill_tokens"] != requests * w.inputs
                or self.counters["decode_tokens"] != requests * (w.outputs - 1)
                or len(self.tokens) != requests * w.outputs
            ):
                raise SimulationStateError(
                    "serving request or buffer conservation failed"
                )
            self.bank.validate_empty()
            return dict(
                seconds=self.now,
                requests=requests,
                input_tokens=requests * w.inputs,
                output_tokens=requests * w.outputs,
                decode_forwards=requests * (w.outputs - 1),
                output_tps=requests * w.outputs / self.now,
                requests_per_s=requests / self.now,
                **latency_metrics(self.tokens, self.requests),
                counters=dict(self.counters),
                maximum_reserved_input_slots=self.max_pending,
                request_timings=self.requests,
                profile_count=len(self.profiles),
                resource_stats={
                    k: dict(zip(("busy_s", "wait_s", "jobs"), v))
                    for k, v in self.kernel.stats().items()
                },
                scope="Complete finite cohorts from empty KV through the final output; stage entry copies, bounded input slots, cache order and token feedback are explicit. No arrival/network/tokenization service or LP30 compiler execution is calibrated.",
            )
        finally:
            self.kernel.close()
