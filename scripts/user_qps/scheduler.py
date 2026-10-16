"""Poisson arrivals and round-robin decode with per-user FIFO queues."""

from collections import deque
import random
from types import SimpleNamespace


def arrival_times(qps, count, seed):
    rng = random.Random(seed)
    times, now = [], 0.0
    for _ in range(count):
        now += rng.expovariate(qps)
        times.append(now)
    return times


def workload(qps, count, users, seed, unique_users=False):
    """Aggregate Poisson process with uniform or one-request-per-user marks."""
    rng = random.Random(seed + 1)
    if unique_users:
        if count > users:
            raise ValueError("unique user assignment requires requests <= users")
        marks = list(range(users))
        rng.shuffle(marks)
        marks = marks[:count]
    else:
        marks = [rng.randrange(users) for _ in range(count)]
    return [
        dict(request_id=i, user=marks[i], arrival_s=t)
        for i, t in enumerate(arrival_times(qps, count, seed))
    ]


def schedule(
    env,
    arrivals,
    *,
    users,
    max_batch,
    output_tokens,
    context,
    serve,
    request_rows,
    token_rows,
    batch_rows,
    queue_rows,
    busy_policy="queue",
    progress=None,
):
    """One active request/user; per-user FIFO, round-robin decode tokens.

    QPS is independent of completions. All queued requests drain after the final
    arrival. A new request may arrive during an iteration; admission is at the
    next iteration boundary. Users with no arrived requests remain idle.
    """
    pending = [deque() for _ in range(users)]
    busy = [False] * users
    runnable = deque()
    wake = env.event()
    arrivals_done = False

    def snapshot(event):
        queue_rows.append(
            dict(
                time_s=env.now,
                event=event,
                pending=sum(map(len, pending)),
                active_users=sum(busy),
                runnable=len(runnable),
            )
        )

    def admit(row, ready):
        busy[row["user"]] = True
        row["eligible_s"] = ready
        runnable.append(
            SimpleNamespace(
                id=row["request_id"],
                slot=row["user"],
                is_prefill=False,
                prefill_len=context,
                context_len=context,
                generation_idx=0,
                ready=ready,
                last_token=None,
                row=row,
            )
        )

    def produce():
        nonlocal arrivals_done, wake
        for source in arrivals:
            yield env.timeout(source["arrival_s"] - env.now)
            row = dict(source, status="accepted")
            request_rows.append(row)
            user = row["user"]
            if busy[user]:
                if busy_policy == "drop":
                    row["status"] = "dropped"
                else:
                    pending[user].append(row)
            else:
                admit(row, env.now)
            snapshot("arrival")
            if not wake.triggered:
                wake.succeed()
        arrivals_done = True
        if not wake.triggered:
            wake.succeed()

    def consume():
        nonlocal wake
        iteration = 0
        while True:
            if not runnable:
                if arrivals_done:
                    break
                wake = env.event()
                yield wake
                continue
            requests = [
                runnable.popleft() for _ in range(min(max_batch, len(runnable)))
            ]
            assert len({r.slot for r in requests}) == len(requests)
            start = env.now
            for req in requests:
                req.row.setdefault("first_admitted_s", start)
            snapshot("dispatch")
            completions = yield env.process(serve(requests, iteration))
            batch_rows.append(
                dict(
                    iteration=iteration,
                    start_s=start,
                    end_s=env.now,
                    batch_size=len(requests),
                    users=[r.slot for r in requests],
                    request_ids=[r.id for r in requests],
                )
            )
            for req in requests:
                end = completions[req.id]
                assert req.ready <= start + 1e-9 and start <= end <= env.now + 1e-9
                token_rows.append(
                    dict(
                        user=req.slot,
                        request_id=req.id,
                        token_index=req.generation_idx,
                        context_len=req.context_len,
                        ready_s=req.ready,
                        admitted_s=start,
                        completion_s=end,
                        scheduler_wait_ms=(start - req.ready) * 1000,
                        service_ms=(end - start) * 1000,
                        itl_ms=(
                            None
                            if req.last_token is None
                            else (end - req.last_token) * 1000
                        ),
                    )
                )
                req.row.setdefault("first_token_s", end)
                req.generation_idx += 1
                if req.generation_idx == output_tokens:
                    req.row.update(
                        status="complete", completion_s=end, output_tokens=output_tokens
                    )
                    if pending[req.slot]:
                        row = pending[req.slot].popleft()
                        admit(row, max(end, row["arrival_s"]))
                    else:
                        busy[req.slot] = False
                else:
                    req.context_len = context + req.generation_idx
                    req.ready = req.last_token = end
                    runnable.append(req)
            snapshot("iteration_complete")
            iteration += 1
            if progress is not None and (iteration == 1 or iteration % 100 == 0):
                progress(iteration, sum(busy), sum(map(len, pending)))
        assert not any(pending) and not any(busy)

    env.process(produce())
    env.process(consume())
