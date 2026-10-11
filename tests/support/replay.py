"""A controlled completion clock for workload dependency tests."""

from TokenSim.timing import TimingRecorder


class ControlledEngine:
    def __init__(self, env, latency=0.1, cancel_delay=0):
        self.env = env
        self.latency = latency
        self.cancel_delay = cancel_delay
        self.submitted = []
        self.timing = TimingRecorder()

    def add_requests(self, requests):
        for request in requests:
            self.submitted.append(request)
            self.env.process(self.complete(request))

    def complete(self, request):
        yield self.env.timeout(self.latency)
        if request.cancel_requested:
            return
        request.record_prefill_admission(0, 0)
        for _ in range(request.decode_len):
            request.step(self.env, 0, 1, self.timing)

    def cancel_requests(self, requests):
        events = []
        for request in requests:
            events.append(request.request_cancel(self.env))
            self.env.process(self.cancel(request))
        return events

    def cancel(self, request):
        yield self.env.timeout(self.cancel_delay)
        request.acknowledge_cancel(self.env)

    def reset_profile_stats(self):
        self.timing.reset()
