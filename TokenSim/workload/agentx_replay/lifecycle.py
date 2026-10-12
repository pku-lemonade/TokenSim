"""Request submission, completion, branch activation and cancellation."""

from __future__ import annotations

import uuid
import simpy
from TokenSim.llm.llm_request import Request
from .records import engine_hashes, TraceTurn


class RequestLifecycle:
    """Request submission, completion, branch activation and cancellation."""

    def _joins_ready(self, play, cursor):
        if cursor.conversation is not play.graph.root:
            return True
        return all(
            b.branch_id in play.opened_branches
            and all(play.streams[cid].terminal for cid in b.child_conversation_ids)
            for b in play.graph.branches
            if b.join_turn_index == cursor.index
        )

    def _ready(self, play, cursor):
        if (
            not cursor.active
            or cursor.terminal
            or cursor.in_flight
            or cursor.index >= len(cursor.conversation.turns)
        ):
            return False
        turn = cursor.conversation.turns[cursor.index]
        return (
            cursor.due is not None
            and cursor.due <= self.env.now + 1e-9
            and self._joins_ready(play, cursor)
            and all(p in play.completed for p in turn.predecessors)
        )

    def _make_request(self, play, turn, *, warmup):
        req = Request(
            self._next_id,
            turn.request.input_length,
            1 if warmup else turn.requested_output_tokens,
            self.block_size,
            chat_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"agentx:{self.workload_seed}:{play.graph.trace_id}:{play.pass_index}:{play.lane}:{turn.key.conversation_id}",
                )
            ),
            parent_chat_id=(
                None if turn.source_kind == "weka_main" else play.graph.trace_id
            ),
            turn=turn.key.turn_index,
            hash_ids=engine_hashes(
                turn.request.hash_ids,
                trace_block_size=play.graph.block_size,
                engine_block_size=self.block_size,
                input_tokens=turn.request.input_length,
            ),
            cache_salt=play.scope,
            reuse_group=play.graph.trace_id,
            measurement_phase="warmup" if warmup else "profile",
            record_timing=not warmup,
        )
        self._next_id += 1
        req.source = TraceTurn(
            trace_id=play.graph.trace_id,
            play_id=play.pass_index,
            lane=play.lane,
            stream_id=(
                "root" if turn.source_kind == "weka_main" else turn.key.conversation_id
            ),
            conversation_id=turn.key.conversation_id,
            outer_index=turn.source.outer_index,
            inner_index=turn.source.inner_index,
            recorded_model=turn.request.model,
        )
        req.trace_hash_ids = turn.request.hash_ids
        req.trace_block_size = play.graph.block_size
        return req

    def _submit(self, req, play, cid=None, *, primer=False):
        phase = "warmup" if req.measurement_phase == "warmup" else "profiling"
        req.arrive(self.env)
        self._in_flight[req.id] = (req, play, cid, primer)
        self.phase_counts[phase]["sent"] += 1
        self.trace_ids_used[phase].add(req.source.trace_id)
        self.source_ids_used[phase].add(
            (
                req.source.trace_id,
                req.source.outer_index,
                req.source.inner_index,
            )
        )
        if not primer:
            play.in_flight += 1
            play.idle_started_at = None
            play.idle_cap_expired = False
            play.streams[cid].in_flight = req
        self._global_idle_started = None
        req.completion_event.callbacks.append(
            lambda event, rid=req.id: self._completed(rid, event)
        )
        submit = getattr(self.engine, "add_requests_burst", self.engine.add_requests)
        submit([req])
        (self.warmup_requests if phase == "warmup" else self.requests).append(req)
        self._event(
            "issued",
            request_id=req.id,
            lane=play.lane,
            conversation_id=req.source.conversation_id,
            primer=primer,
        )

    def _completed(self, request_id, event):
        wire = self._in_flight.pop(request_id, None)
        if wire is None:
            return
        req, play, cid, primer = wire
        phase = "warmup" if req.measurement_phase == "warmup" else "profiling"
        if not event.ok:
            event.defused = True
        outcome = (
            "cancelled"
            if req.cancel_requested
            else "successful" if event.ok and req.is_done else "error"
        )
        self.phase_counts[phase][outcome] += 1
        self._record_terminal(req, play, outcome)
        if not primer:
            cursor = play.streams[cid]
            play.in_flight -= 1
            cursor.in_flight = None
            if outcome != "cancelled" or phase == "warmup":
                old_turn = cursor.conversation.turns[cursor.index]
                play.completed.add(old_turn.key)
                cursor.index += 1
                if cursor.index >= len(cursor.conversation.turns):
                    play.set_terminal(cursor)
                due = (
                    self.env.now
                    if self.phase == "pressure"
                    else (
                        self._next_due(
                            play, cursor, cursor.conversation.turns[cursor.index]
                        )
                        if not cursor.terminal
                        else None
                    )
                )
                self._set_due(cursor, due)
                if cursor.conversation is play.graph.root:
                    self._open_branches(play, old_turn)
            else:
                play.set_terminal(cursor)
                cursor.due = None
            if play.in_flight == 0:
                play.idle_started_at = self.env.now
        if not self._in_flight:
            self._global_idle_started = self.env.now
        req.release_logical_blocks()
        req.hash_ids = None
        self._notify()

    def _record_terminal(self, req, play, outcome):
        phase = "warmup" if req.measurement_phase == "warmup" else "profiling"
        source = req.first_prefill_source
        record = {
            "request_id": req.id,
            "phase": phase,
            "outcome": outcome,
            "start_s": req.arrival_at,
            "end_s": self.env.now,
            "input_tokens": req.prefill_len,
            "gpu_hit_tokens": source.gpu_hit_tokens if source else None,
            "cpu_hit_tokens": source.cpu_hit_tokens if source else None,
            "computed_input_tokens": source.computed_tokens if source else None,
            "prefill_source_observed": source is not None,
            "prefill_admission_count": req.prefill_admission_count,
            "reloaded_gpu_tokens": req.reloaded_gpu_tokens,
            "reloaded_cpu_tokens": req.reloaded_cpu_tokens,
            "reloaded_generated_gpu_tokens": req.reloaded_generated_gpu_tokens,
            "reloaded_generated_cpu_tokens": req.reloaded_generated_cpu_tokens,
            "latest_gpu_context_hit_tokens": req.gpu_cached_context_tokens,
            "latest_cpu_context_hit_tokens": req.external_cached_context_tokens,
            "recomputation_count": req.recomputation_count,
            "recomputed_tokens": req.recomputed_tokens_total,
            "latest_gpu_hit_tokens": req.cached_prefill_tokens
            - req.external_cached_prefill_tokens,
            "latest_cpu_hit_tokens": req.external_cached_prefill_tokens,
            "latest_computed_input_tokens": req.effective_prefill_tokens,
            "requested_output_tokens": req.decode_len,
            "output_tokens": req.generation_idx,
            "trace_id": req.source.trace_id,
            "conversation_id": req.source.conversation_id,
            "outer_index": req.source.outer_index,
            "inner_index": req.source.inner_index,
            "turn_index": req.turn,
            "lane": play.lane,
            "dp_rank": req.dp_rank,
            "worker_id": req.routing_worker_id,
            "pass_index": play.pass_index,
            "cancellation_acknowledged": (
                req.cancelled_at is not None if req.cancel_requested else None
            ),
            "length_capped": req.max_model_len is not None
            and req.context_len >= req.max_model_len,
            "ttft_s": req.prefill_latency if req.generation_idx else None,
            "latency_s": (
                req.completed_at - req.arrival_at
                if req.completed_at is not None
                else None
            ),
        }
        if self.request_sink:
            self.request_sink(record)
        else:
            self.records.append(record)

    def _open_branches(self, play, completed_turn):
        for branch in play.graph.branches:
            if (
                branch.spawning_turn_index != completed_turn.key.turn_index
                or branch.branch_id in play.opened_branches
            ):
                continue
            play.opened_branches.add(branch.branch_id)
            for cid in branch.child_conversation_ids:
                cursor = play.streams[cid]
                cursor.active = True
                if not cursor.conversation.turns:
                    play.set_terminal(cursor)
                    continue
                if not cursor.terminal:
                    play.active_streams[cid] = cursor
                if self.phase == "pressure":
                    self._set_due(cursor, self.env.now)
                elif self.issue_mode == "recorded":
                    self._set_due(
                        cursor,
                        self._next_due(
                            play, cursor, cursor.conversation.turns[cursor.index]
                        ),
                    )
                else:
                    offset = max(
                        0.0,
                        cursor.conversation.turns[cursor.index].request.t
                        - completed_turn.request.end_s,
                    )
                    self._set_due(cursor, self.env.now + offset)

    def _recycle(self):
        for lane, play in enumerate(self.trees):
            if not play.finished() or not self.accepting:
                continue
            if (
                self.phase == "pressure"
                and self.pressure_sent[lane] >= self.warmup_quota
            ):
                continue
            graph = self.pool.next_graph()
            self.recycle_trace_ids[lane].append(graph.trace_id)
            self.trees[lane] = self._tree(graph, lane)

    def _issue_ready(self):
        if not self.accepting:
            return False
        self._recycle()
        issued = False
        limit = self.env.now + 1e-9
        for play in self.trees:
            # A ready stream has a due timer and no request in flight.
            if not any(
                cursor.in_flight is None
                and cursor.due is not None
                and cursor.due <= limit
                for cursor in play.active_streams.values()
            ):
                continue
            for cid, cursor in sorted(
                play.active_streams.items(), key=lambda item: item[1].ordinal
            ):
                if (
                    self.phase == "pressure"
                    and self.pressure_sent[play.lane] >= self.warmup_quota
                ):
                    break
                if not self._ready(play, cursor):
                    continue
                req = self._make_request(
                    play,
                    cursor.conversation.turns[cursor.index],
                    warmup=self.phase == "pressure",
                )
                if self.phase == "pressure":
                    self.pressure_sent[play.lane] += 1
                self._submit(req, play, cid)
                issued = True
        return issued

    def _primer(self, offset, play, key, epoch):
        yield self.env.timeout(offset)
        if epoch != self._epoch or not self.accepting:
            return
        turn = play.streams[key.conversation_id].conversation.turns[key.turn_index]
        req = self._make_request(play, turn, warmup=True)
        self._submit(req, play, primer=True)
        failure_event = getattr(self.engine, "failure_event", None)
        yield (
            simpy.events.AnyOf(self.env, [req.completion_event, failure_event])
            if failure_event is not None
            else req.completion_event
        )
        if hasattr(self.engine, "raise_if_failed"):
            self.engine.raise_if_failed()

    def _drain(self, deadline):
        while self._in_flight and self.env.now < deadline:
            yield from self._wait(deadline)
        if self._in_flight:
            pending = [entry[0] for entry in self._in_flight.values()]
            self.engine.cancel_requests(pending)
            end = self.env.now + self.cancellation_timeout
            while self._in_flight and self.env.now < end:
                yield from self._wait(end)
            self.cancel_unreturned += len(self._in_flight)
            for req, play, cid, primer in list(self._in_flight.values()):
                phase = "warmup" if req.measurement_phase == "warmup" else "profiling"
                self.phase_counts[phase]["cancelled"] += 1
                self.phase_counts[phase]["cancel_unreturned"] += 1
                self._record_terminal(req, play, "cancelled")
                if phase == "warmup" and not primer:
                    cursor = play.streams[cid]
                    if cursor.index < len(cursor.conversation.turns):
                        old_turn = cursor.conversation.turns[cursor.index]
                        play.completed.add(old_turn.key)
                        cursor.index += 1
                        cursor.in_flight = None
                        if cursor.index >= len(cursor.conversation.turns):
                            play.set_terminal(cursor)
                        play.in_flight -= 1
                        if cursor.conversation is play.graph.root:
                            self._open_branches(play, old_turn)
            self._in_flight.clear()
