"""Token counts and resource timing with retained or online event accounting."""

from collections import defaultdict

from TokenSim.config.validation import number
from TokenSim.errors import SimulationStateError
from TokenSim.af.trace.occupancy import (
    occupied_until,
    validate_resource_capacity,
    unique_executions,
    resource_queue_intervals,
)
from TokenSim.af.serving.timing import namespace_resource


class ResourceMetrics:
    def __init__(self, window, *, namespace=""):
        self.window = window
        self.namespace = namespace
        self.busy = defaultdict(float)
        self.queue = defaultdict(float)
        self.counts = defaultdict(int)
        self._execution_groups = {}

    def _resource(self, name):
        return namespace_resource(self.namespace, name) if self.namespace else name

    def observe(self, event):
        group = getattr(event, "execution_group_id", None)
        if group is not None:
            # Online spans are emitted at completion, before simulation time
            # advances. Keep only the latest shared invocation per device.
            resources = event.resource_ids or (event.resource_id,)
            if self._execution_groups.get(resources) == group:
                return
            self._execution_groups[resources] = group
        primary = self._resource(event.resource_id)
        if "start" not in self.window:
            self.queue[primary] += 0.0
            return
        start = self.window["start"]
        end = self.window.get("end", event.end_at)
        queued = sum(
            max(0.0, min(end, right) - max(start, left))
            for left, right in resource_queue_intervals(event)
        )
        self.queue[primary] += queued
        overlap = max(0.0, min(end, occupied_until(event)) - max(start, event.start_at))
        if not overlap:
            return
        for name in event.resource_ids or (event.resource_id,):
            resource = self._resource(name)
            self.busy[resource] += overlap
            self.counts[resource] += 1

    def merge_disjoint(self, other):
        for target, source in (
            (self.busy, other.busy),
            (self.queue, other.queue),
            (self.counts, other.counts),
        ):
            if target.keys() & source.keys():
                raise SimulationStateError(
                    "online serving metrics require disjoint service resource namespaces"
                )
            target.update(source)

    def summary(self, events, *, capacities=None, include_end=False):
        start, end = self.window["start"], self.window["end"]
        number(start, "measurement start")
        number(end, "measurement end", minimum=start, positive=True)
        if end == start:
            raise ValueError("steady-state measurement must have positive duration")
        selected = [
            row
            for row in events
            if not row.get("warmup", False)
            and start <= row["time"]
            and (row["time"] <= end if include_end else row["time"] < end)
        ]
        output = sum(row["tokens"] for row in selected)
        decode = sum(row["tokens"] for row in selected if row["phase"] == "decode")
        seconds = end - start
        return {
            "start_s": start,
            "end_s": end,
            "duration_s": seconds,
            "output_tokens": output,
            "decode_tokens": decode,
            "output_tokens_per_s": output / seconds,
            "decode_tokens_per_s": decode / seconds,
            "resource_utilization": {
                resource: value / (seconds * (capacities or {}).get(resource, 1))
                for resource, value in self.busy.items()
            },
            "resource_queue_s": dict(self.queue),
            "resource_operation_counts": dict(self.counts),
        }


def steady_metrics(events, traces, *, start, end, capacities=None, include_end=False):
    traces = list(traces)
    validate_resource_capacity(traces, capacities or {})
    resources = ResourceMetrics(dict(start=start, end=end))
    for event in unique_executions(traces):
        resources.observe(event)
    return resources.summary(events, capacities=capacities, include_end=include_end)
