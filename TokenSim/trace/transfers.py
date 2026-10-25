"""Observe synchronous plans and queued storage DMA without changing timing."""

from TokenSim.kv_transfer.connectors.base import flatten_connectors
from TokenSim.kv_transfer.native_storage import QueuedKVLink
from TokenSim.trace.events import TraceEvent
from TokenSim.trace.execution import observe_service


class TransferObserver:
    def __init__(self, session):
        self.session = session
        self.links = {}
        self.pending = {}

    def attach(self, worker):
        connector = worker.connector
        for child in flatten_connectors(connector):
            for direction in ("read", "write"):
                link = getattr(child, direction + "_link", None)
                if link is not None:
                    self._attach_link(link, worker.id, direction)

    def _attach_link(self, link, worker_id, direction):
        if isinstance(link, QueuedKVLink):
            if id(link) in self.links:
                self.links[id(link)]["workers"].add(worker_id)
                self.links[id(link)]["directions"].add(direction)
                return
            binding = dict(
                resource=f"kv:link:{len(self.links)}",
                workers={worker_id},
                directions={direction},
            )
            self.links[id(link)] = binding
            original = link.transfer

            def transfer(size):
                index = self.session.iteration
                self.session.iteration += 1
                job = dict(
                    iteration=index,
                    ready=link.env.now,
                    resource=binding["resource"],
                    worker=(
                        next(iter(binding["workers"]))
                        if len(binding["workers"]) == 1
                        else None
                    ),
                    size=size,
                    direction=(
                        next(iter(binding["directions"]))
                        if len(binding["directions"]) == 1
                        else "shared"
                    ),
                    run_phase=self.session.run_phase,
                )
                self.pending[index] = job
                try:
                    yield from observe_service(
                        original(size), lambda: job.update(start=link.env.now)
                    )
                except BaseException:
                    job["stopped_at"] = link.env.now
                    raise
                if "start" in job:
                    self.session.recorder.record(self._event(job, link.env.now))
                    self.session._complete(index)
                self.pending.pop(index, None)

            link.transfer = transfer
            return
        for name in ("host", "network"):
            child = getattr(link, name, None)
            if child is not None:
                self._attach_link(
                    child, None if name == "network" else worker_id, direction
                )

    @staticmethod
    def _event(job, end, *, incomplete=False):
        return TraceEvent(
            iteration_id=job["iteration"],
            block_id=0,
            layer_id=-1,
            phase="mixed",
            event_type="transfer",
            resource_id=job["resource"],
            ready_at=job["ready"],
            start_at=job["start"],
            end_at=end,
            queue_duration=job["start"] - job["ready"],
            service_duration=end - job["start"],
            request_count=0,
            token_count=0,
            source_id="modeled:queued_kv_link",
            match_type="analytical",
            worker_id=job["worker"],
            operation="kv_dma",
            node_id=f"kv:dma:{job['iteration']}",
            run_phase=job["run_phase"],
            payload_bytes=None if incomplete else job["size"],
            incomplete=incomplete,
            details=dict(
                direction=job["direction"],
                planned_payload_bytes=job["size"],
                correlation_scope="resource_only",
                incomplete=incomplete,
            ),
        )

    def partial_events(self):
        for job in self.pending.values():
            if "start" in job:
                end = min(
                    self.session.engine.env.now,
                    job.get("stopped_at", self.session.engine.env.now),
                )
                if end > job["start"]:
                    yield self._event(job, end, incomplete=True)
