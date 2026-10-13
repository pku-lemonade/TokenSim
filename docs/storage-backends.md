# Storage backend plugins

Uniform Mooncake stores accept an optional storage backend through the `offload` configuration object.
The core manages cache objects, admission, least-recently-used eviction, write readiness and read pinning.
The backend owns its configuration, capacity and media service-time estimates.
Existing solid-state drive (SSD) options remain supported through a compatibility adapter.

## Selecting a backend

Put the following object inside `kv_connector_extra_config` for `MooncakeStoreConnector`.

```json
{
  "offload": {
    "backend": "my_storage_package:ExampleBackend",
    "name": "flash",
    "config": {
      "capacity_bytes": 1099511627776
    }
  }
}
```

`backend` selects an installed Python module's exported class or a registered builder alias.
The module reference uses `module:class` syntax.
Only the configured module is imported; the core does not discover or import optional backend packages at startup.
An unavailable package produces a configuration error identifying its reference.

`name` identifies this backend instance within the shared store and defaults to `offload`.
The name `memory` is reserved for the store's host-memory tier.
`config` contains backend-owned options and defaults to an empty object.
The selected builder receives a copy of that object and validates its fields.
Its reported capacity must be a positive integer byte count when offload is enabled.

Providing `offload` enables the backend.
Combine it with the existing host-memory and connector options, including `memory_capacity_blocks`, namespaces and save policies.
Legacy `ssd_*` fields and `enable_offload: false` cannot be combined with a plugin selection.
For an existing SSD configuration, keep its `enable_offload`, `offload_tier: "ssd"` and `ssd_*` fields unchanged.
The built-in `ssd` builder also accepts the generic format with `capacity_bytes`, `read_bandwidth_bytes_per_s`, `write_bandwidth_bytes_per_s`, `read_latency_us` and `write_latency_us` in `config`.

## Implementing a backend

Import the public contracts from `TokenSim.storage`.
A builder accepts one configuration dictionary and returns an object satisfying `StorageBackend`.

```python
from TokenSim.config.validation import integer, number
from TokenSim.storage import IOCost, IORequest


class ExampleBackend:
    def __init__(self, config):
        if set(config) - {"capacity_bytes", "bandwidth_bytes_per_s"}:
            raise ValueError("unknown storage option")
        self.capacity_bytes = integer(config["capacity_bytes"], "capacity_bytes", minimum=1)
        self.bandwidth = number(config["bandwidth_bytes_per_s"], "bandwidth_bytes_per_s", positive=True)

    def estimate_io(self, request: IORequest) -> IOCost:
        return IOCost(request.nbytes / self.bandwidth)

    def describe(self) -> dict:
        return {
            "capacity_bytes": self.capacity_bytes,
            "bandwidth_bytes_per_s": self.bandwidth,
        }
```

Production backends must reject non-finite numeric inputs and unknown options.
The [HBF example package](../examples/storage/hbf/) provides a complete configuration validator and operation-size bandwidth curve.

`IORequest` contains `operation` (`read` or `write`), `nbytes` and `access_pattern` (`auto`, `random` or `sequential`).
The current uniform store sends `auto` requests with the bytes in each stored object.
The backend returns `IOCost(service_seconds)` with a finite, nonnegative media service time.
Network transfer and cache ownership remain outside this estimate.
The core does not call the backend for zero-byte operations and rejects costs that do not use `IOCost`.

`describe()` returns a stable JSON object containing normalized configuration values and defaults.
Clients sharing a Mooncake master must select compatible backend references, instance names and descriptions.
The constructor and description must remain deterministic and must not create simulator processes or modify global state.
The core may construct a backend while validating configuration and comparing shared-store settings.

Python applications can register a builder before parsing their connector configuration.

```python
from TokenSim.storage import StorageBackendFactory

StorageBackendFactory.register_backend("example", ExampleBackend)
```

That application can select `"backend": "example"`.
Duplicate aliases are rejected, and `ssd` is reserved for the built-in builder.
Command-line configurations can use a module reference without application startup registration.

## Results and ownership

`storage_metrics` groups counters by Mooncake master and backend instance name.

```json
{
  "storage_metrics": {
    "default": {
      "flash": {
        "backend": "my_storage_package:ExampleBackend",
        "profile": {
          "capacity_bytes": 1099511627776,
          "bandwidth_bytes_per_s": 107374182400
        },
        "hit_count": 1,
        "read_operations": 1,
        "read_blocks": 1,
        "read_bytes": 4096,
        "read_service_seconds": 0.00000003814697265625,
        "write_operations": 1,
        "write_blocks": 1,
        "write_bytes": 4096,
        "write_service_seconds": 0.00000003814697265625
      }
    }
  }
}
```

The profile comes from the backend's `describe()` method.
Each positive-byte media operation increments the corresponding operation, block, byte and service-time counters.
A lookup with an offload hit increments `hit_count` once.
Counter aggregation preserves master and instance identity, and warmup reset clears counters while retaining profiles.
Legacy SSD configurations continue to populate `mooncake_ssd_*` counters and their existing offload profile.
Plugin configurations use the generic dictionary and do not add media-specific fields to the result schema.

Store objects become reusable after their transfer and media write finish.
Reads pin their objects through the modeled completion time, and the common offload tier applies the same capacity and eviction rules to every backend.
The current interface retains the uniform store's blocking I/O timing and does not add media queue contention or multi-tier migration.
Grouped KV pages continue to use the native DRAM page store and reject offload backend selections.
