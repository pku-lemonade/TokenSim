"""Forward-query geometry derived from one scheduler admission."""

from dataclasses import dataclass

from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class AFRequestWork:
    queries: int
    offset: int
    context: int
    logits_rows: int
    forward_phase: str


def request_work(request, phase):
    context = int(request.context_len)
    building = bool(request.is_prefill or getattr(request, "needs_recompute", False))
    default_queries = (
        int(getattr(request, "prefill_compute_len", request.prefill_len))
        if building
        else 1
    )
    queries = int(
        getattr(
            request, "query_tokens", getattr(request, "step_tokens", default_queries)
        )
    )
    if building:
        offset = int(getattr(request, "num_computed_tokens", context - queries))
        if request.is_prefill and not getattr(request, "needs_recompute", False):
            offset = max(
                offset,
                int(getattr(request, "cached_prefill_tokens", 0)),
                context - default_queries,
            )
    else:
        queries = int(
            getattr(request, "scheduled_query_tokens", 0)
            or getattr(request, "scheduled_tokens", 0)
            or 1
        )
        offset = max(0, context - queries)
    if queries < 1 or offset < 0 or offset + queries > context:
        raise ConfigurationError(
            "AF scheduled queries must lie inside the request context"
        )
    return AFRequestWork(
        queries,
        offset,
        offset + queries,
        int(offset + queries == context) if building else queries,
        "prefill" if building else phase,
    )
