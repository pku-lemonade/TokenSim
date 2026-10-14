"""Complete ready-input operation costs, distinct from CPU/device busy time."""

from collections.abc import Mapping
from TokenSim.af.calibration import content_fingerprint, _bounds, _nonnegative
from TokenSim.af.estimate_cache import freeze
from TokenSim.errors import ConfigurationError

CONTROL_OPERATIONS = {
    "h2d": ("control_h2d_us", "control_staging"),
    "length_readback": ("control_length_readback_us", "control_length_readback"),
    "payload_readback": ("control_payload_readback_us", "control_payload_readback"),
}
A_PREPARATION_OPERATIONS = ("a_input_prepare", "a_attention_metadata")
LEGACY_OPERATIONS = {
    **CONTROL_OPERATIONS,
    "a_input_prepare": ("a_step_prepare_us", "a_input_prepare"),
    "a_attention_metadata": ("a_step_prepare_us", "a_attention_metadata"),
    "f_dp_metadata": ("f_step_prepare_us", "f_dp_metadata"),
}
OPERATIONS = {
    **LEGACY_OPERATIONS,
    "sampled_token_readback": (None, "sampled_token_readback"),
}


class OperationServiceProfile:
    def __init__(
        self, document, *, expected_runtime, expected_environment, explicit_costs
    ):
        self.fingerprint = content_fingerprint(document)
        if not isinstance(document, Mapping):
            raise ConfigurationError("operation profile must be an object")
        required = {
            "schema_version",
            "profile_id",
            "timing_scope",
            "unit",
            "runtime_fingerprint",
            "environment",
            "batch_range",
            "context_range",
            "payload_bytes",
            "operations",
            "evidence",
        }
        if set(document) != required:
            raise ConfigurationError("operation profile fields mismatch")
        d = self.document = freeze(document)
        if (
            type(d["schema_version"]) is not int
            or d["schema_version"] not in (1, 2, 3, 4)
            or d["timing_scope"] != "operation_service"
            or d["unit"] != "us"
        ):
            raise ConfigurationError("unsupported operation profile scope/version/unit")
        if not isinstance(d["profile_id"], str) or not d["profile_id"]:
            raise ConfigurationError("operation profile requires identity")
        if (
            d["runtime_fingerprint"] != expected_runtime
            or not expected_environment
            or d["environment"] != expected_environment
        ):
            raise ConfigurationError("operation profile runtime/environment mismatch")
        self.batch_range = _bounds(d["batch_range"], "batch_range")
        self.context_range = _bounds(d["context_range"], "context_range")
        if type(d["payload_bytes"]) is not int or d["payload_bytes"] <= 0:
            raise ConfigurationError(
                "operation profile requires positive wire payload bytes"
            )
        ops = d["operations"]
        supported = LEGACY_OPERATIONS if d["schema_version"] == 1 else OPERATIONS
        if not isinstance(ops, Mapping) or not ops or set(ops) - set(supported):
            raise ConfigurationError("unsupported operation")
        for name, entry in ops.items():
            if not isinstance(entry, Mapping) or set(entry) != {
                "fixed_us",
                "coverage",
                "input_ready",
                "completion",
                "resource_model",
            }:
                raise ConfigurationError("operation entry fields mismatch")
            _nonnegative(entry["fixed_us"], name)
            if not isinstance(entry["coverage"], (list, tuple)) or tuple(
                entry["coverage"]
            ) != (OPERATIONS[name][1],):
                raise ConfigurationError("operation coverage mismatch")
            completion = (
                "device_synchronize_after_operation"
                if name in A_PREPARATION_OPERATIONS
                else "cpu_api_return_data_ready"
            )
            ready = (
                "sampled_tokens_device_ready"
                if name == "sampled_token_readback"
                else (
                    "control_decoded_cpu_metadata_ready"
                    if name == "f_dp_metadata"
                    else "prior_device_work_complete"
                )
            )
            if (
                entry["input_ready"] != ready
                or entry["completion"] != completion
                or entry["resource_model"] != "calling_thread_single_inflight"
            ):
                raise ConfigurationError(
                    "unsupported operation ready/completion/resource boundary"
                )
            if OPERATIONS[name][0] in explicit_costs:
                raise ConfigurationError("duplicate operation/manual coverage: " + name)
        e = d["evidence"]
        if not isinstance(e, Mapping) or set(e) != {
            "source",
            "kind",
            "external_wait_excluded",
            "audit",
            "method",
            "windows",
            "training_batches",
            "heldout_batches",
            "validation_status",
            "statistics",
        }:
            raise ConfigurationError("operation evidence fields mismatch")
        if (
            e["kind"] not in ("measured", "synthetic_test")
            or e["external_wait_excluded"] is not True
            or e["method"] != "equal_window_medians"
        ):
            raise ConfigurationError("operation evidence must exclude external waiting")
        if any(not isinstance(e[k], str) or not e[k] for k in ("source", "audit")):
            raise ConfigurationError(
                "operation evidence requires source and boundary audit"
            )
        if (
            type(e["windows"]) is not int
            or e["windows"] < 3
            or not isinstance(e["training_batches"], (list, tuple))
            or not isinstance(e["heldout_batches"], (list, tuple))
            or tuple(e["training_batches"]) != (1, 8)
            or tuple(e["heldout_batches"])
            != {1: (4,), 2: (2,), 3: (5,), 4: (7,)}[d["schema_version"]]
        ):
            raise ConfigurationError(
                "operation evidence requires three windows and versioned held-out batch"
            )
        if self.batch_range != (1, 8) or e["validation_status"] not in (
            "pending",
            "heldout_passed",
            "heldout_failed",
        ):
            raise ConfigurationError("unsupported operation workload/validation range")
        if not isinstance(e["statistics"], Mapping) or set(e["statistics"]) != set(ops):
            raise ConfigurationError("operation statistics must cover each operation")
        for name, stats in e["statistics"].items():
            if not isinstance(stats, Mapping) or set(stats) != {
                "p50_us",
                "p95_us",
                "samples",
            }:
                raise ConfigurationError("operation statistics fields mismatch")
            if (
                type(stats["samples"]) is not int
                or stats["samples"] < 1
                or _nonnegative(stats["p50_us"], name)
                > _nonnegative(stats["p95_us"], name)
            ):
                raise ConfigurationError("invalid operation statistics")

    def validate_workload(self, requests, phase):
        from TokenSim.af.control import encode_eager_control

        if (
            phase != "decode"
            or not self.batch_range[0] <= len(requests) <= self.batch_range[1]
            or any(
                not self.context_range[0] <= r.context_len <= self.context_range[1]
                for r in requests
            )
        ):
            raise ConfigurationError("workload outside operation profile scope")
        if len(encode_eager_control(len(requests))) != self.document["payload_bytes"]:
            raise ConfigurationError("operation wire size mismatch")

    def describe(self):
        return dict(
            profile_id=self.document["profile_id"],
            fingerprint=self.fingerprint,
            timing_scope="operation_service",
            operations=self.document["operations"],
            evidence_kind=self.document["evidence"]["kind"],
            status="operation_profile_loaded_not_full_engine_validation",
        )
