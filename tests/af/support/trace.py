"""Observation counters used by trace regression tests."""

from dataclasses import replace

from TokenSim.af.planning.batching import AFIteration
from TokenSim.af.trace.metrics import AFTraceEvent
from TokenSim.af.trace.recorder import AFTraceRecorder


def _event(
    node_id,
    operation,
    ready,
    start,
    end,
    *,
    predecessors=(),
    resource="worker:0",
    payload=None,
    details=None,
):
    return AFTraceEvent(
        iteration_id=7,
        block_id=0,
        layer_id=0,
        phase="decode",
        event_type="compute" if operation.startswith("compute") else "transfer",
        operation=operation,
        resource_id=resource,
        resource_ids=(resource,),
        ready_at=ready,
        start_at=start,
        end_at=end,
        queue_duration=start - ready,
        service_duration=end - start,
        request_count=1,
        token_count=1,
        source_id="test",
        node_id=node_id,
        predecessor_ids=predecessors,
        payload_bytes=payload,
        details=details or {},
    )


def _valid_events():
    return (
        _event("c0", "compute_a", 0.0, 0.0, 1.0, resource="worker:0"),
        _event("c1", "compute_a", 0.0, 0.0, 2.0, resource="worker:1"),
        _event(
            "send",
            "af_a_to_f",
            2.0,
            2.0,
            3.0,
            predecessors=("c0", "c1"),
            resource="link:a_to_f",
            payload=32,
            details={"payload_bytes": 32},
        ),
        _event(
            "return",
            "af_f_to_a",
            3.0,
            3.0,
            4.0,
            predecessors=("send",),
            resource="link:f_to_a",
            payload=16,
            details={"payload_bytes": 16},
        ),
    )


class AccessList(list):
    visits = 0

    def __iter__(self):
        for item in super().__iter__():
            self.visits += 1
            yield item

    def __getitem__(self, index):
        self.visits += 1
        return super().__getitem__(index)


def lookup_visits(steps):
    recorder = AFTraceRecorder()
    recorder.events = AccessList()
    for step in range(steps):
        for event in _valid_events():
            recorder.record(replace(event, iteration_id=step))
        recorder.events_for(step)
        recorder.validate_invariants(step)
        recorder.summarize(AFIteration(step, "decode", (), (), 0, 4))
    return dict(events=len(recorder.events), visits=recorder.events.visits)
