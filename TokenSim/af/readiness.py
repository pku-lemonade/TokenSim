"""Per-invocation communication readiness; never part of a service estimate."""
from dataclasses import dataclass

import simpy

from TokenSim.errors import SimulationStateError


@dataclass(frozen=True)
class TransferReadiness:
    """Events completing when each stream/input and the receive call are ready.

    Callers create these events for the current invocation. A host API return
    alone does not make a CUDA stream ready. Omitted events mean readiness is
    already covered by the DAG; this assumption is exported in the trace.
    """

    sender_stream: simpy.Event | None = None
    receiver_stream: simpy.Event | None = None
    receiver_posted: simpy.Event | None = None

    def events(self, env):
        result = []
        for event in (self.sender_stream, self.receiver_stream, self.receiver_posted):
            if event is None:
                continue
            if not isinstance(event, simpy.Event) or event.env is not env:
                raise SimulationStateError("transfer readiness must use events from the current simulation")
            if event not in result:
                result.append(event)
        return tuple(result)
