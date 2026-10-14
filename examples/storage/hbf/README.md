# Optional HBF storage backend

This independently installable package implements high-bandwidth flash (HBF) media estimates through the public `TokenSim.storage` contracts.
Its parameter validation, default values and bandwidth curve are contained in `tokensim_hbf/`.
The TokenSim core has no dependency on this package.

Install the package from the repository root after installing TokenSim.

```bash
python -m pip install -e examples/storage/hbf
```

Select `tokensim_hbf:HBFBackend` in the connector's `offload.backend` field.
See the [HBF configuration guide](../../../docs/hbf.md) for runnable examples and the [storage plugin contract](../../../docs/storage-backends.md) for implementing another backend.

Capacity uses bytes, bandwidth uses bytes per second and fixed media latency uses microseconds.
Random 4 KiB bandwidth defaults to 80% of sequential bandwidth.
For automatic access patterns, effective bandwidth interpolates logarithmically between 4 KiB and 10 MiB and stays at the endpoint values outside that interval.
The default fixed read/write latencies are 12/120 microseconds.
These are simulation assumptions carried from the public HBF snapshot.
