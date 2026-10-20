"""Completion-driven clients, open arrivals and request-level measurements."""

import numpy as np
import simpy

from TokenSim.config.validation import number
from TokenSim.errors import ConfigurationError


def distribution(values):
    return {
        name: float(np.percentile(values, q)) if values else None
        for name, q in (("p50", 50), ("p95", 95), ("p99", 99))
    } | {
        "mean": float(np.mean(values)) if values else None,
        "count": len(values),
    }


class Arrivals:
    def __init__(
        self, env, begin, stop, measurement, *, qps=None, pattern="uniform", seed=0
    ):
        if qps is not None:
            number(qps, "arrival QPS", positive=True)
        if pattern not in {"uniform", "poisson"}:
            raise ConfigurationError("arrival pattern must be uniform or poisson")
        self.env, self.stop, self.measurement = env, stop, measurement
        self.qps, self.pattern = qps, pattern
        self.queue = simpy.Store(env)
        self.emitted, self.admitted = [], []
        self.rng = np.random.default_rng(seed)
        if qps is not None:
            env.process(self.source(begin))

    def source(self, begin):
        yield begin
        while not self.stop.triggered:
            interval = (
                1 / self.qps
                if self.pattern == "uniform"
                else self.rng.exponential(1 / self.qps)
            )
            yield self.env.timeout(interval)
            if self.stop.triggered:
                return
            self.emitted.append(self.env.now)
            yield self.queue.put(self.env.now)

    def acquire(self):
        if self.qps is None:
            self.emitted.append(self.env.now)
            arrival = self.env.now
        else:
            pending = self.queue.get()
            done = yield pending | self.stop
            if pending not in done:
                pending.cancel()
                return None
            arrival = done[pending]
        if self.stop.triggered:
            return None
        self.admitted.append((arrival, self.env.now))
        return arrival

    def summary(self):
        start, end = self.measurement["start"], self.measurement["end"]
        emitted = [t for t in self.emitted if start <= t <= end]
        admitted = [(a, t) for a, t in self.admitted if start <= t <= end]
        samples = []
        for t in np.linspace(start, end, 9):
            samples.append(
                dict(
                    time_s=float(t),
                    pending=int(
                        sum(a <= t for a in emitted) - sum(b <= t for _, b in admitted)
                    ),
                )
            )
        return dict(
            target_qps=self.qps,
            pattern=self.pattern,
            arrivals=len(emitted),
            admissions=len(admitted),
            arrival_qps=len(emitted) / (end - start),
            admission_wait_s=distribution([t - a for a, t in admitted]),
            pending_at_end=len(emitted) - len(admitted),
            backlog_samples=samples,
            backlog_growth_per_s=(samples[-1]["pending"] - samples[4]["pending"])
            / ((end - start) / 2),
        )


def request_metrics(turns, workload, clients, micro_batch, measurement):
    start, end = measurement["start"], measurement["end"]
    rows, occupied = [], 0.0
    for turn in turns:
        if turn["warmup"]:
            continue
        size = min(micro_batch, clients - turn["cohort"] * micro_batch)
        occupied += size * max(0, min(turn["end"], end) - max(turn["start"], start))
        if not turn["completed"] or turn["start"] < start or turn["end"] > end:
            continue
        first = turn["start"] + turn["ttft_s"]
        submitted = turn.get("submitted_at", turn["start"])
        elapsed = turn["end"] - submitted
        generated = turn["end"] - first
        for index in range(size):
            rows.append(
                dict(
                    user=turn["cohort"] * micro_batch + index,
                    round=turn["round"],
                    submitted_at=submitted,
                    admitted_at=turn["start"],
                    first_output_at=first,
                    last_output_at=turn["end"],
                    output_tokens=workload.output_tokens,
                    generation_tps=(workload.output_tokens - 1) / generated,
                    request_tps=workload.output_tokens / elapsed,
                    ttft_s=first - submitted,
                    admission_wait_s=turn["start"] - submitted,
                    service_latency_s=turn["end"] - turn["start"],
                    latency_s=elapsed,
                    tpot_s=generated / (workload.output_tokens - 1),
                )
            )
    duration = end - start
    result = {
        key: distribution([r[key] for r in rows])
        for key in (
            "generation_tps",
            "request_tps",
            "ttft_s",
            "latency_s",
            "tpot_s",
            "service_latency_s",
        )
    }
    result.update(
        completed_requests=len(rows),
        achieved_qps=len(rows) / duration,
        mean_active_users=occupied / duration,
        active_user_seconds=occupied,
        little_law_complete_request_users=sum(r["latency_s"] for r in rows) / duration,
        little_law_complete_active_users=sum(r["service_latency_s"] for r in rows)
        / duration,
        requests=rows,
    )
    return result
