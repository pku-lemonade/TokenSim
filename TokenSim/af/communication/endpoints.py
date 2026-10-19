"""Independent endpoint injection budgets with atomic multi-endpoint admission."""

import simpy

from TokenSim.errors import ConfigurationError
from TokenSim.config.validation import number


class EndpointResources:
    def __init__(self, env, networks, identities, *, transfer_occupancy="service"):
        from TokenSim.af.communication.resources import DomainScheduler

        self.env, self.networks, self.identities = env, networks, set(identities)
        if transfer_occupancy not in {"service", "wire"}:
            raise ConfigurationError(
                "endpoint transfer occupancy must be service or wire"
            )
        self.transfer_occupancy = transfer_occupancy
        self.slots = {}
        self.scheduler = DomainScheduler(env, ())

    def for_step(self, network, direction, endpoints):
        if network not in self.networks or not set(endpoints) <= self.identities:
            raise ConfigurationError(
                "endpoint transport requires declared networks and mapped workers"
            )
        collective = direction in {
            "collective",
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
            "tp_collective",
        }
        owners = (
            [(endpoint, leg) for endpoint in endpoints for leg in ("tx", "rx")]
            if collective
            else [(endpoints[0], "tx"), (endpoints[-1], "rx")]
        )
        keys = tuple(
            sorted(
                {
                    ("endpoint", f"{self.networks[network]}:{endpoint}:{leg}")
                    for endpoint, leg in owners
                }
            )
        )
        for key in keys:
            self.slots.setdefault(key, simpy.Resource(self.env, capacity=1))
        self.scheduler.resource_ids = self.scheduler.resource_ids.union(
            self.format_key(key) for key in keys
        )
        return tuple((key, self.slots[key]) for key in keys)

    @staticmethod
    def format_key(key):
        return f"link:{key[0]}:{key[1]}"

    def capacities(self):
        return {self.format_key(key): 1 for key in self.slots}

    def run_transfer(self, keys, duration, propagation, identity):
        number(duration, "transfer duration")
        number(propagation, "transfer hop latency", maximum=duration)
        flight = propagation if self.transfer_occupancy == "wire" else 0.0
        ready, start, injected = yield from self.scheduler.run(
            keys, duration - flight, identity
        )
        if flight:
            yield self.env.timeout(flight)
        return ready, start, self.env.now, injected
