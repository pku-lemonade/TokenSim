from __future__ import annotations

from TokenSim.llm.llm_request import LLMTime, RequestTime


class TimingRecorder:
    """Completion timing owned by one simulation engine."""

    def __init__(self, timing: LLMTime | None = None):
        self.timing = timing if timing is not None else LLMTime()

    def record_request(self, request_time: RequestTime) -> None:
        self.timing.time.append(request_time)

    def reset(self) -> LLMTime:
        self.timing.time.clear()
        return self.timing
