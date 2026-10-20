"""Explicit token-ID messages between external draft and target pools."""


def token_control(
    services,
    communication,
    hardware,
    events,
    workload,
    bundle,
    cohort,
    round_index,
    step,
    reverse=False,
):
    draft_owner = services["draft"].owners[bundle["draft"][0].dp_rank]
    target_owner = services["target"].owners[bundle["target"][0].dp_rank]
    pair = (
        (target_owner.physical_worker, draft_owner.physical_worker)
        if reverse
        else (draft_owner.physical_worker, target_owner.physical_worker)
    )
    size = len(bundle["target"]) * 4 * (1 if reverse else workload.draft_tokens)
    keys = tuple(
        communication.format_key(key)
        for key, _ in communication.for_step("handoff", "a_to_f", pair)
    )
    link = hardware.link("ethernet1600Gb")
    times = yield from communication.run_transfer(
        keys,
        link.analytical_transfer_us(size) / 1e6,
        link.latency_us / 1e6,
        (cohort, round_index, step, int(reverse)),
    )
    events.append(
        dict(
            source=pair[0],
            target=pair[1],
            bytes=size,
            network="handoff",
            ready_s=times[0],
            start_s=times[1],
            end_s=times[2],
            injection_end_s=times[3],
        )
    )
