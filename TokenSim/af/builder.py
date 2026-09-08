from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from TokenSim.af.models import AFBlock, AFBlockPlacement, AFIteration, AFPhase
from TokenSim.config.cluster_config import AFLinkConfig
from TokenSim.errors import ConfigurationError


def phase_for_requests(requests: Sequence[Any]) -> AFPhase:
    if not requests:
        raise ConfigurationError("AF iteration requires at least one request")
    prefill = [bool(request.is_prefill) for request in requests]
    if all(prefill):
        return "prefill"
    if not any(prefill):
        return "decode"
    raise ConfigurationError(
        "AF iteration requires a homogeneous Prefill or Decode batch"
    )


class AFBlockBuilder:
    def build(
        self,
        iteration_id: int,
        requests: Sequence[Any],
        af_block_size: int,
        source_a_worker_id: int,
        link: AFLinkConfig,
        phase: AFPhase | None = None,
        placement_factory: (
            Callable[[int, tuple[Any, ...]], AFBlockPlacement] | None
        ) = None,
        histogram_factory: Callable[[tuple[Any, ...]], dict[int, int]] | None = None,
    ) -> AFIteration:
        resolved_phase = phase or phase_for_requests(requests)
        if type(af_block_size) is not int or af_block_size < 1:
            raise ConfigurationError("af_block_size must be a positive integer")
        if resolved_phase not in link.phases:
            raise ConfigurationError(
                f"AF link {link.name!r} does not cover phase {resolved_phase!r}"
            )

        blocks: list[AFBlock] = []
        for block_id, start in enumerate(range(0, len(requests), af_block_size)):
            block_requests = tuple(requests[start : start + af_block_size])
            if resolved_phase == "prefill":
                token_count = sum(
                    int(
                        getattr(
                            request,
                            "prefill_compute_len",
                            getattr(request, "prefill_len"),
                        )
                    )
                    for request in block_requests
                )
            else:
                token_count = len(block_requests)
            blocks.append(
                AFBlock(
                    iteration_id=iteration_id,
                    block_id=block_id,
                    phase=resolved_phase,
                    request_ids=tuple(int(request.id) for request in block_requests),
                    request_count=len(block_requests),
                    token_count=token_count,
                    context_lengths=tuple(
                        int(request.context_len) for request in block_requests
                    ),
                    source_a_worker_id=source_a_worker_id,
                    link_name=link.name,
                    placement=(
                        placement_factory(block_id, block_requests)
                        if placement_factory is not None
                        else None
                    ),
                    expert_histogram=tuple(
                        sorted(
                            (
                                histogram_factory(block_requests)
                                if histogram_factory
                                else {}
                            ).items()
                        )
                    ),
                )
            )
        return AFIteration(
            iteration_id=iteration_id,
            phase=resolved_phase,
            blocks=tuple(blocks),
            request_ids=tuple(int(request.id) for request in requests),
        )
