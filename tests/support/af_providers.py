"""Af providers."""

from __future__ import annotations

from TokenSim.af import AFServiceEstimate


class FixedComputeProvider:
    def __init__(
        self,
        a: float,
        f: float,
        *,
        router: float | None = None,
        expert: float | None = None,
    ):
        router = f / 3 if router is None else router
        expert = f - router if expert is None else expert
        self.durations = {
            "a": a,
            "router": router,
            "expert": expert,
            "attention": a,
            "dense_ffn": f,
            "routed_expert": expert,
            "shared_expert": expert,
            "weighted_combine": 0.0,
            "residual": 0.0,
            "layout": 0.0,
            "encode": 0.0,
            "decode": 0.0,
        }
        self.queries = []

    def estimate(self, query):
        self.queries.append(query)
        return AFServiceEstimate(
            service_duration=self.durations[query.stage],
            source_id="test-stub:compute",
            details={"stage": query.stage},
        )


class FixedTransferProvider:
    def __init__(
        self,
        a_to_f: float,
        f_to_a: float,
        dispatch: float = 0.0,
        combine: float = 0.0,
        collective: float = 0.0,
    ):
        self.durations = {"a_to_f": a_to_f, "f_to_a": f_to_a}
        self.operation_durations = {
            "ep_dispatch": dispatch,
            "ep_combine": combine,
            "tp_collective": collective,
            "all_reduce": collective,
            "all_gather": collective,
            "reduce_scatter": collective,
        }
        self.queries = []

    def estimate(self, query):
        self.queries.append(query)
        duration = self.operation_durations.get(query.operation)
        if duration is None:
            duration = self.durations[query.direction]
        return AFServiceEstimate(
            service_duration=duration,
            source_id="test-stub:transfer",
            payload_bytes=query.payload_bytes,
            details={"direction": query.direction, "operation": query.operation},
        )


class InvalidComputeProvider(FixedComputeProvider):
    def __init__(self, invalid_duration):
        super().__init__(invalid_duration, 0.0)


class UnsupportedComputeProvider:
    def estimate(self, query):
        raise LookupError(f"no sample for {query.device_type}:{query.stage}")
