"""Recorded dispatch timers, idle-gap compression and replay wakeups."""

from __future__ import annotations

import heapq
import math
import simpy
from TokenSim.errors import SimulationStateError
from .records import REPLAY_PROTOCOL, engine_hashes, TraceTurn, StreamCursor, TreePlay


class ReplayClock:
    """Recorded dispatch timers, idle-gap compression and replay wakeups."""

    def _recorded_due(self, origin_s, request_s, clock_origin_s=None):
        # A recycled tree starts its own source clock at the recycle time.
        anchor = self.env.now if clock_origin_s is None else clock_origin_s
        return anchor + max(0.0, request_s - origin_s)

    def _set_due(self, cursor, due):
        cursor.due = due
        if self.issue_mode == "recorded" and due is not None:
            self._recorded_timer_sequence += 1
            heapq.heappush(
                self._recorded_timer_heap,
                (due, self._recorded_timer_sequence, cursor),
            )

    def _register_play_timers(self, play):
        if self.issue_mode != "recorded":
            return
        for cursor in play.active_streams.values():
            if cursor.due is not None:
                self._set_due(cursor, cursor.due)

    def _next_recorded_timer(self):
        while self._recorded_timer_heap:
            due, _, cursor = self._recorded_timer_heap[0]
            if (
                cursor.terminal
                or not cursor.active
                or cursor.in_flight is not None
                or cursor.due != due
                or due <= self.env.now + 1e-9
            ):
                heapq.heappop(self._recorded_timer_heap)
                continue
            return due
        return None

    def _next_due(self, play, cursor, turn):
        if self.issue_mode == "recorded":
            return self._recorded_due(
                play.origin_s, turn.request.t, play.clock_origin_s
            )
        return self.env.now + turn.wait_after_previous_s

    def _future_timers(self, play):
        # ``active_streams`` holds only active, non-terminal cursors.
        limit = self.env.now + 1e-9
        return [
            cursor
            for cursor in play.active_streams.values()
            if cursor.in_flight is None
            and cursor.due is not None
            and cursor.due > limit
        ]

    def _idle_shifts(self):
        if (
            self.phase != "profiling"
            or not self.accepting
            or self.issue_mode == "recorded"
        ):
            return
        all_timers = []
        for play in self.trees:
            timers = self._future_timers(play)
            if (
                timers
                and play.in_flight == 0
                and play.idle_started_at is not None
                and self.tree_idle_cap is not None
                and (
                    play.idle_cap_expired
                    or self.env.now >= play.idle_started_at + self.tree_idle_cap
                )
            ):
                shift = min(c.due for c in timers) - self.env.now
                for cursor in timers:
                    cursor.due -= shift
                play.idle_cap_expired = True
                self.tree_idle_jumps += 1
                self.tree_idle_seconds_skipped += shift
                self._event("tree_idle_shift", lane=play.lane, skipped_s=shift)
                # The shift made some timers due now.
                timers = self._future_timers(play)
            all_timers.extend(timers)
        timers = all_timers
        if (
            timers
            and not self._in_flight
            and self._global_idle_started is not None
            and self.system_idle_cap is not None
            and self.env.now >= self._global_idle_started + self.system_idle_cap
        ):
            shift = min(c.due for c in timers) - self.env.now
            for cursor in timers:
                cursor.due -= shift
            self.system_idle_jumps += 1
            self.system_idle_seconds_skipped += shift
            self._event("system_idle_shift", skipped_s=shift)

    def _wake_at(self, boundary, timers_by_play):
        if not self.accepting:
            return boundary
        if self.issue_mode == "recorded":
            next_due = self._next_recorded_timer()
            return min(boundary, next_due) if next_due is not None else boundary
        candidates = [boundary]
        for play, timers in zip(self.trees, timers_by_play):
            if (
                self.phase == "pressure"
                and self.pressure_sent[play.lane] >= self.warmup_quota
            ):
                continue
            candidates.extend(c.due for c in timers)
            if (
                timers
                and self.phase == "profiling"
                and play.in_flight == 0
                and self.tree_idle_cap is not None
                and play.idle_started_at is not None
            ):
                candidates.append(
                    self.env.now
                    if play.idle_cap_expired
                    else max(self.env.now, play.idle_started_at + self.tree_idle_cap)
                )
        if (
            self.phase == "profiling"
            and not self._in_flight
            and any(timers_by_play)
            and self.system_idle_cap is not None
            and self._global_idle_started is not None
        ):
            candidates.append(
                max(self.env.now, self._global_idle_started + self.system_idle_cap)
            )
        return min(candidates)

    def _wait(self, boundary):
        (
            self.engine.raise_if_failed()
            if hasattr(self.engine, "raise_if_failed")
            else None
        )
        # Timers do not change while this method decides; collect them once.
        timers_by_play = (
            []
            if self.issue_mode == "recorded"
            else [self._future_timers(play) for play in self.trees]
        )
        has_future_timer = (
            self._next_recorded_timer() is not None
            if self.issue_mode == "recorded"
            else any(timers_by_play)
        )
        if self.accepting and not self._in_flight and not has_future_timer:
            raise SimulationStateError(
                "AgentX graph has no ready request, future timer, or in-flight predecessor"
            )
        wake_at = self._wake_at(boundary, timers_by_play)
        changed = self._changed
        events = [changed]
        failure_event = getattr(self.engine, "failure_event", None)
        if failure_event is not None:
            events.append(failure_event)
        if math.isfinite(wake_at):
            if (
                not self._cached_wake
                or self._cached_wake[0] != wake_at
                or self._cached_wake[1].triggered
            ):
                self._cached_wake = (
                    wake_at,
                    self.env.timeout(max(0, wake_at - self.env.now)),
                )
            events.append(self._cached_wake[1])
        elif not self._in_flight:
            raise SimulationStateError(
                "AgentX dependency graph stalled without an in-flight request or future timer"
            )
        yield simpy.events.AnyOf(self.env, events)
