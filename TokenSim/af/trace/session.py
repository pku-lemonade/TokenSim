"""Attach AF producer boundaries to an engine's global recording session."""

from dataclasses import replace

from TokenSim.trace.events import TraceEvent


def attach_af_session(engine, session):
    executor = engine.af_executor
    if executor is None:
        return
    owners = {}
    original_submit = executor.submit
    original_record = executor.trace.record
    original_finish = engine.record_af_iteration

    def submit(worker, requests):
        requests = tuple(requests)
        iteration = original_submit(worker, requests)
        ids = tuple(request.id for request in requests)
        dependencies = dict(
            session.request_tails[r] for r in ids if r in session.request_tails
        )
        owners[iteration.iteration_id] = worker, ids, dependencies, session.run_phase
        return iteration

    def record(event):
        owner = owners.get(event.iteration_id)
        if owner is not None:
            details = dict(event.details, run_phase=owner[3])
            if not event.predecessor_ids:
                details["external_predecessors"] = owner[2]
            event = replace(event, details=details, run_phase=owner[3])
        original_record(event)

    def finish(iteration):
        original_finish(iteration)
        owner = owners.pop(iteration.iteration_id, None)
        if owner is None:
            return
        worker, ids, _, run_phase = owner
        events = executor.trace.events_for(iteration.iteration_id)
        ends = {
            event.node_id: event.end_at
            for event in events
            if abs(event.end_at - iteration.end_at) < 1e-12
        }
        node = f"af:completion:{iteration.iteration_id}"
        session.recorder.record(
            TraceEvent(
                iteration_id=session.iteration,
                block_id=worker,
                layer_id=-1,
                phase="decode",
                event_type="synchronize",
                resource_id=f"af:completion:{worker}",
                ready_at=iteration.end_at,
                start_at=iteration.end_at,
                end_at=iteration.end_at,
                service_duration=0,
                queue_duration=0,
                request_count=len(ids),
                token_count=0,
                source_id="af_execution",
                operation="af_completion",
                worker_id=worker,
                node_id=node,
                predecessor_ids=tuple(ends),
                details={"request_ids": ids, "run_phase": run_phase},
                run_phase=run_phase,
            )
        )
        session._complete(session.iteration, ends)
        session.iteration += 1
        for request_id in ids:
            session.request_tails[request_id] = node, iteration.end_at

    executor.submit = submit
    executor.trace.record = record
    engine.record_af_iteration = finish
