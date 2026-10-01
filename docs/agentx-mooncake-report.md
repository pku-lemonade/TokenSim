# Mooncake backend status

The unified AgentX runner now accepts `--backend mooncake` and writes a per-run Mooncake configuration. The hybrid scheduler supports synchronous external load/save latency in DP lockstep, and the Mooncake store has a bulk in-memory admission path for long prompt page chains.

The native DSV4 hybrid manager does not yet have the asynchronous shared-page completion adapter required by the reference Mooncake connector. Full native attempts therefore use an explicit `admission_policy=never`, synchronous control configuration so an unfinished async save cannot be reported as a successful comparison. The control still exercises connector construction, key identity, service configuration and synchronous lifecycle checks, but it is not a valid shared-cache accuracy result. Longer full-window control attempts were stopped after confirming the current synchronous path is too slow for the one-hour matrix.

The production SimpleCPU full windows and all native operator stages remain valid. Mooncake shared-store comparison stays open until its async page import/release adapter is implemented; no Mooncake throughput or hit-rate number is included in the accuracy summary.
