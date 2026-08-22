from __future__ import annotations

from TokenSim.llm.llm_request import LLMTime, RequestTime, g_time


class TimingRecorder:
    """Facade for request completion timing.

    The recorder keeps the existing `g_time` storage compatible while giving
    request/worker code an explicit recording boundary.
    """

    def __init__(self, timing: LLMTime | None = None):
        self.timing = timing if timing is not None else g_time

    def record_request(self, request_time: RequestTime) -> None:
        self.timing.time.append(request_time)

    def reset(self) -> LLMTime:
        self.timing.time.clear()
        return self.timing


DEFAULT_TIMING_RECORDER = TimingRecorder(g_time)


def reset_timing() -> LLMTime:
    return DEFAULT_TIMING_RECORDER.reset()
