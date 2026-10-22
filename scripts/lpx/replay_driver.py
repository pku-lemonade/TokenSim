"""Run unchanged serving protocols with sampled native-DAG execution profiles."""

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import simpy

from TokenSim.af.communication.endpoints import EndpointResources
from TokenSim.af.serving.metrics import ResourceMetrics
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError, SimulationStateError
from scripts.lpx.native_metrics import resource_capacities
from scripts.lpx.replay_engine import ReplayKernel, ReplayNode
from scripts.lpx.replay_profiles import cached_profile

ROOT = Path(__file__).resolve().parents[2]


class ReplayEnvironment(simpy.Environment):
    def __init__(self, driver):
        super().__init__()
        self.driver = driver

    def run(self, until=None):
        from simpy.core import StopSimulation

        if until is not None:
            if not isinstance(until, simpy.Event):
                raise ConfigurationError(
                    "sampled replay requires an Event run boundary"
                )
            if until.processed:
                return until.value
            until.callbacks.append(StopSimulation.callback)
        try:
            result = self._run_events()
            if until is not None and not until.processed:
                raise RuntimeError("replay exhausted before its run boundary")
            return result
        except StopSimulation as exc:
            return exc.args[0]

    def _run_events(self):
        while self._queue or self.driver.kernel.active:
            self.driver.sync_window()
            limit = self.peek()
            priority = (1 if self._queue[0][1] == 0 else 3) if self._queue else 4
            result = self.driver.kernel.advance(limit, priority)
            if result is not None:
                self._now = result.end
                event = self.driver.pending.pop(result.id)
                event.succeed(result)
            elif self._queue:
                self.step()
            elif self.driver.kernel.active:
                raise SimulationStateError("replay stalled without an external event")


class ReplayEndpoints(EndpointResources):
    def __init__(self, driver, networks, identities, occupancy):
        super().__init__(driver.env, networks, identities, transfer_occupancy=occupancy)
        self.driver = driver
        self.profiles = {}

    def register_profile_endpoint(self, name):
        prefix = "link:endpoint:"
        value = name[len(prefix) :]
        network, owner = value.split(":", 1)
        physical, leg = owner.rsplit(":", 1)
        if (
            network not in self.networks.values()
            or physical not in self.identities
            or leg not in {"tx", "rx"}
        ):
            raise ConfigurationError(
                "native replay profile references an undeclared endpoint"
            )
        key = ("endpoint", value)
        self.slots.setdefault(key, simpy.Resource(self.env, capacity=1))

    def run_transfer(self, keys, duration, propagation, identity):
        occupancy = duration - (propagation if self.transfer_occupancy == "wire" else 0)
        signature = (tuple(keys), duration, occupancy)
        if signature not in self.profiles:
            self.profiles[signature] = self.driver.kernel.add(
                [ReplayNode((), tuple(keys), keys[0], 0, duration, occupancy)]
            )
        result = yield self.driver.submit(self.profiles[signature])
        return result.ready, result.start, result.end, result.released


@dataclass(frozen=True)
class ReplayConfig:
    workload: object
    samples: int = 4
    prefill_samples: int = 1
    cache: str = str(ROOT / "outputs/lpx-replay-profile-cache")
    streaming: dict | None = None

    def __call__(self, runtimes, measurement):
        integer(self.samples, "decode profile samples", minimum=1)
        integer(self.prefill_samples, "prefill profile samples", minimum=1)
        if self.streaming is not None:
            from TokenSim.comm.streaming import StreamingConfig

            StreamingConfig(**self.streaming)
        return ReplayDriver(self, runtimes, measurement)


class ReplayDriver:
    def __init__(self, config, runtimes, measurement):
        self.config, self.runtimes, self.measurement = config, runtimes, measurement
        self.kernel = ReplayKernel()
        self.pending, self.profiles = {}, {}
        self.env = ReplayEnvironment(self)
        self.names, self.services, self.metadata = {}, {}, []

    def communication(self, networks, identities, occupancy):
        return ReplayEndpoints(self, networks, identities, occupancy)

    def bind(self, services, communication):
        self.services, self.endpoints = services, communication
        self.names = {id(service): name for name, service in services.items()}

    def sync_window(self):
        if "start" in self.measurement:
            self.kernel.window(
                self.measurement["start"], self.measurement.get("end", math.inf)
            )

    def submit(self, profile):
        self.sync_window()
        job = self.kernel.submit(profile, self.env.now)
        event = self.env.event()
        self.pending[job] = event
        return event

    def profile(self, service, requests, phase):
        namespace = self.names[id(service)]
        request = requests[0]
        queries = request.scheduled_query_tokens
        batch = len(requests)
        dp = service.sessions[request.id].dp_rank
        workload = self.config.workload
        context = request.context_len
        if workload.cached_tokens + workload.new_tokens >= 16384:
            context = (
                workload.cached_tokens
                + (
                    workload.new_tokens
                    if request.emit_logits
                    else workload.new_tokens // 2
                )
                if phase == "prefill"
                else workload.cached_tokens
                + workload.new_tokens
                + max(queries, (workload.output_tokens - 1) // 2)
            )
        context = max(queries, context)
        count = (
            self.config.prefill_samples if phase == "prefill" else self.config.samples
        )
        selected = (
            int(
                np.random.SeedSequence(
                    [
                        workload.seed,
                        request.id,
                        request.generation_idx,
                        request.context_len,
                        queries,
                        int(phase == "prefill"),
                    ]
                ).generate_state(1)[0]
            )
            % count
        )
        signature = (
            namespace,
            batch,
            queries,
            phase,
            request.emit_logits,
            context,
            dp,
            selected,
        )
        if signature not in self.profiles:
            profile = cached_profile(
                service.runtime,
                namespace,
                self.config.cache,
                batch=batch,
                queries=queries,
                phase=phase,
                emit_logits=request.emit_logits,
                context=context,
                dp_rank=dp,
                seed=(workload.seed + selected * 104729),
                sample=selected,
                **(
                    {"streaming": self.config.streaming}
                    if self.config.streaming is not None
                    else {}
                ),
            )
            capacities = resource_capacities(self.services)
            for node in profile.nodes:
                for key in node.resources:
                    if key.startswith("link:endpoint:"):
                        self.endpoints.register_profile_endpoint(key)
                    elif capacities.get(key) != 1:
                        raise ConfigurationError(
                            "sampled replay requires declared unit-capacity resources"
                        )
            self.profiles[signature] = self.kernel.add(profile.nodes)
            self.metadata.append(profile.metadata)
        return self.profiles[signature]

    def forward(self, service, requests, phase):
        if len({request.context_len for request in requests}) != 1:
            raise ConfigurationError(
                "sampled replay requires synchronized request cohorts"
            )
        result = yield self.submit(self.profile(service, requests, phase))
        return SimpleNamespace(iteration_id=result.id)

    def metrics(self, token_events):
        metrics = ResourceMetrics(self.measurement)
        for name, (busy, waited, count) in self.kernel.stats().items():
            metrics.queue[name] = waited
            if count:
                metrics.busy[name], metrics.counts[name] = busy, count
        return metrics.summary(
            token_events,
            capacities=resource_capacities(self.services),
            include_end=True,
        )

    def describe(self):
        result = dict(
            method=(
                "modeled_native_operator_dags"
                if self.config.streaming is not None
                else "sampled_native_operator_dags"
            ),
            decode_profile_samples=self.config.samples,
            prefill_profile_samples=self.config.prefill_samples,
            profiles=self.metadata,
            routing="deterministic selection from a finite native routing corpus",
            context="exact below 16K input tokens; representative midpoint above 16K",
            limitations=[
                "Sampled routing and fixed within-phase context are approximations.",
                "Resource tie ordering uses native sample start order and replay submission order.",
                "All requested tokens and serving protocol actions are executed; model quality is unmeasured.",
            ],
            **(
                {"streaming": self.config.streaming}
                if self.config.streaming is not None
                else {}
            ),
        )
        self.kernel.close()
        return result
