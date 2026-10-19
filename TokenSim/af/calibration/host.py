"""Strict host-only profiles. No runtime queue or device launch is fitted here."""

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math

from TokenSim.af.caching.estimates import freeze
from TokenSim.errors import ConfigurationError


HOST_COVERAGE = {
    "a_step_prepare_us": "a_input_prepare",
    "f_step_prepare_us": "f_loop_prepare",
    "compute_submit_us": "compute_dispatch_excluding_launch",
    "host_submit_us": "send_submit",
    "host_receive_submit_us": "receive_submit",
    "host_completion_us": "data_completion",
    "control_encode_us": "control_encode",
    "control_decode_us": "control_decode",
}
COMPUTE_CALL_FIELDS = {
    "attention": "compute_a_attention_us",
    "post_attention_norm": "compute_a_post_norm_us",
    "router_topk": "compute_f_router_us",
    "routed_expert": "compute_f_expert_group_us",
    "embedding_lookup": "compute_a_embedding_us",
    "final_norm": "compute_a_final_norm_us",
    "lm_head": "compute_a_lm_head_us",
    "token_selection": "compute_a_token_selection_us",
}
MESSAGE_CALL_FIELDS = {
    (backend, direction, action): f"{action}_{backend}_{direction}_us"
    for backend, directions in (
        ("pynccl", ("a_to_f", "f_to_a")),
        ("torch_distributed_nccl", ("a_to_f",)),
    )
    for direction in directions
    for action in ("send", "receive")
}
INVOCATION_REPLACES = {
    **{field: "compute_submit_us" for field in COMPUTE_CALL_FIELDS.values()},
    "compute_f_moe_block_us": "compute_submit_us",
    **{
        field: "host_submit_us" if key[2] == "send" else "host_receive_submit_us"
        for key, field in MESSAGE_CALL_FIELDS.items()
    },
}
EXTENDED_HOST_COVERAGE = {
    **{
        field: "native_call:" + field.removesuffix("_us")
        for field in INVOCATION_REPLACES
    },
    "engine_step_schedule_us": "engine_schedule_excluding_input_preparation",
    "engine_output_process_us": "engine_output_excluding_sampling_and_readback",
    "engine_runner_bookkeeping_us": "runner_bookkeeping_excluding_token_readback",
}
# Copy costs belong to device_service and cannot enter a host_service profile.
DEVICE_COST_FIELDS = {
    "control_h2d_us",
    "control_length_readback_us",
    "control_payload_readback_us",
}


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(v) for v in value)
    return value


def content_fingerprint(value):
    try:
        wire = json.dumps(
            _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"profile data must be finite JSON: {exc}") from exc
    return hashlib.sha256(wire.encode()).hexdigest()


def runtime_fingerprint(plan, cluster, workers, hardware):
    from TokenSim.hardware.domains import CommunicationDomainConfig
    from TokenSim.hardware.domains import MODEL_VERSION

    if plan is None or hardware is None:
        raise ConfigurationError(
            "profile validation requires runtime plan and hardware catalog"
        )
    topology = hardware.topology(cluster.topology) if cluster.topology else None
    link_names = set(cluster.networks.values())
    if topology is not None:
        link_names.update(level.link for level in topology.levels)
    return content_fingerprint(
        dict(
            plan=plan,
            devices={w.hardware: hardware.device(w.hardware) for w in workers},
            networks=cluster.networks,
            links=cluster.af.links,
            link_catalog={name: hardware.link(name) for name in link_names},
            topology=topology,
            transfer_topology_mode=cluster.af.transfer_topology_mode,
            protocol=cluster.af.execution_protocol,
            **(
                {
                    "transport_profile": content_fingerprint(
                        cluster.af.transport_service_profile
                    )
                }
                if cluster.af.transport_service_profile is not None
                else {}
            ),
            **(
                {
                    "communication_domains": cluster.af.communication_domains
                    or CommunicationDomainConfig(),
                    "communication_model_version": MODEL_VERSION,
                }
                if cluster.af.transfer_topology_mode == "physical"
                else {}
            ),
            **(
                {"native_call_order": "eager-preposted-v1"}
                if cluster.af.communication_timing.native_call_order_enabled
                else {}
            ),
        )
    )


def _nonnegative(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ConfigurationError(f"{name} must be finite and non-negative")
    return float(value)


def _bounds(value, name):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(v) is not int or v < 1 for v in value)
        or value[0] > value[1]
    ):
        raise ConfigurationError(f"{name} requires an ordered positive integer pair")
    return tuple(value)


class HostOverheadProfile:
    def __init__(
        self, document, *, expected_runtime, expected_environment, explicit_costs
    ):
        if not isinstance(document, Mapping):
            raise ConfigurationError("host profile must be a JSON object")
        self.fingerprint = content_fingerprint(document)
        self.document = freeze(_plain(document))
        d = self.document
        required = {
            "schema_version",
            "profile_id",
            "unit",
            "timing_scope",
            "contains_wait",
            "excludes_device_launch",
            "protocol",
            "runtime_fingerprint",
            "environment",
            "batch_range",
            "context_range",
            "costs",
            "evidence",
        }
        if d.get("schema_version") == 4:
            required |= {"submission_audit", "device_service_source_id"}
        if set(d) != required:
            raise ConfigurationError(
                f"profile fields mismatch: missing={required-set(d)}, unknown={set(d)-required}"
            )
        if (
            type(d["schema_version"]) is not int
            or d["schema_version"] not in (1, 2, 3, 4)
            or d["unit"] != "us"
        ):
            raise ConfigurationError("unsupported host profile version/unit")
        if d["schema_version"] == 4:
            audit = d["submission_audit"]
            if (
                d["timing_scope"] != "host_submission_service"
                or d["contains_wait"] is not False
                or d["excludes_device_launch"] is not False
                or not isinstance(audit, Mapping)
                or audit.get("source_hashes_verified") is not True
                or audit.get("call_counts_verified") is not True
                or audit.get("blocking_apis_absent") is not True
                or audit.get("copy_kinds") != (3,)
                or not audit.get("source")
                or not str(d["device_service_source_id"]).startswith("device_service:")
            ):
                raise ConfigurationError(
                    "CPU CUDA submission requires audited nonblocking APIs and separate device-only calibration"
                )
        elif (
            d["timing_scope"] != "host_service"
            or d["contains_wait"] is not False
            or d["excludes_device_launch"] is not True
        ):
            raise ConfigurationError(
                "host profile must exclude waiting and all existing device/kernel launch costs"
            )
        if d["protocol"] != "vllm_afd" or d["runtime_fingerprint"] != expected_runtime:
            raise ConfigurationError(
                "profile protocol/model/device/topology/runtime fingerprint mismatch"
            )
        if (
            not isinstance(d["environment"], Mapping)
            or not expected_environment
            or dict(d["environment"]) != dict(expected_environment)
        ):
            raise ConfigurationError(
                "profile software/environment fingerprint mismatch or unspecified environment"
            )
        if d["environment"].get("execution_mode") != "eager" or not d[
            "environment"
        ].get("software_sha256"):
            raise ConfigurationError(
                "profile requires eager execution and software_sha256"
            )
        if not isinstance(d["profile_id"], str) or not d["profile_id"]:
            raise ConfigurationError("profile_id must be non-empty")
        self.batch_range = _bounds(d["batch_range"], "batch_range")
        self.context_range = _bounds(d["context_range"], "context_range")
        costs = d["costs"]
        allowed = (
            HOST_COVERAGE
            if d["schema_version"] == 1
            else {**HOST_COVERAGE, **EXTENDED_HOST_COVERAGE}
        )
        if not isinstance(costs, Mapping) or not costs or set(costs) - allowed.keys():
            raise ConfigurationError(
                "host profile costs must be known CPU fields; device copy costs need device_service calibration"
            )
        if set(costs) & set(explicit_costs):
            raise ConfigurationError(
                f"duplicate manual/profile host costs: {sorted(set(costs)&set(explicit_costs))}"
            )
        for field, generic in INVOCATION_REPLACES.items():
            if field in costs and (generic in explicit_costs or generic in costs):
                raise ConfigurationError(
                    f"duplicate native call/generic host coverage: {field}/{generic}"
                )
        coverage = set()
        if "compute_f_moe_block_us" in costs and any(
            name in costs
            for name in ("compute_f_router_us", "compute_f_expert_group_us")
        ):
            raise ConfigurationError(
                "duplicate native F MoE block/router/expert host coverage"
            )
        for name, value in costs.items():
            if not isinstance(value, Mapping) or set(value) != {
                "fixed_us",
                "per_token_us",
                "coverage",
            }:
                raise ConfigurationError(f"{name}: unsupported cost model fields")
            if not isinstance(value["coverage"], (list, tuple)) or tuple(
                value["coverage"]
            ) != (allowed[name],):
                raise ConfigurationError(
                    f"{name}: ambiguous or duplicate operation coverage"
                )
            coverage.update(value["coverage"])
            _nonnegative(value["fixed_us"], name + ".fixed_us")
            _nonnegative(value["per_token_us"], name + ".per_token_us")
            _nonnegative(
                value["fixed_us"] + self.batch_range[1] * value["per_token_us"],
                name + ".maximum_cost",
            )
        evidence = d["evidence"]
        expected = {
            "source",
            "kind",
            "sample_count",
            "window_count",
            "training_batches",
            "heldout_batches",
            "method",
            "residual_us",
            "quantiles_us",
            "validation_status",
        }
        if (
            not isinstance(evidence, Mapping)
            or set(evidence) != expected
            or not isinstance(evidence["source"], str)
            or not evidence["source"]
            or evidence["kind"] not in ("measured", "synthetic_test")
        ):
            raise ConfigurationError(
                "profile requires complete evidence and explicit measured/synthetic_test provenance"
            )
        for field, minimum in (("sample_count", 1), ("window_count", 3)):
            if type(evidence[field]) is not int or evidence[field] < minimum:
                raise ConfigurationError(f"evidence.{field} must be at least {minimum}")
        train, heldout = evidence["training_batches"], evidence["heldout_batches"]
        if (
            not isinstance(train, (list, tuple))
            or not isinstance(heldout, (list, tuple))
            or not train
            or not heldout
            or any(
                type(v) is not int
                or not self.batch_range[0] <= v <= self.batch_range[1]
                for v in (*train, *heldout)
            )
            or set(train) & set(heldout)
        ):
            raise ConfigurationError(
                "profile training/heldout batches must be distinct and within range"
            )
        if evidence["method"] not in ("constant", "nonnegative_linear"):
            raise ConfigurationError("unsupported profile prediction rule")
        if evidence["method"] == "constant" and any(
            v["per_token_us"] != 0 for v in costs.values()
        ):
            raise ConfigurationError("constant profile cannot have a scale coefficient")
        if evidence["method"] == "nonnegative_linear" and len(set(train)) < 3:
            raise ConfigurationError(
                "linear host calibration requires at least three training batches"
            )
        _nonnegative(evidence["residual_us"], "evidence.residual_us")
        quantiles = evidence["quantiles_us"]
        if not isinstance(quantiles, Mapping) or set(quantiles) != set(costs):
            raise ConfigurationError("evidence.quantiles_us must cover each host cost")
        for name, values in quantiles.items():
            if not isinstance(values, Mapping) or set(values) != {"p50", "p95"}:
                raise ConfigurationError("each cost requires p50/p95 service quantiles")
            if _nonnegative(values["p50"], name + ".p50") > _nonnegative(
                values["p95"], name + ".p95"
            ):
                raise ConfigurationError("profile quantiles must be ordered")
        if evidence["validation_status"] not in (
            "pending",
            "heldout_passed",
            "heldout_failed",
        ):
            raise ConfigurationError(
                "profile requires explicit held-out validation status"
            )
        reserved = {1: 4, 2: 2, 3: 5, 4: 7}[d["schema_version"]]
        if reserved in train or reserved not in heldout or 4 in train:
            raise ConfigurationError(
                "native alignment requires versioned held-out validation"
            )

    def validate_workload(self, requests, phase):
        if (
            phase != "decode"
            or not self.batch_range[0] <= len(requests) <= self.batch_range[1]
        ):
            raise ConfigurationError("workload outside host profile phase/batch scope")
        if any(
            not self.context_range[0] <= r.context_len <= self.context_range[1]
            for r in requests
        ):
            raise ConfigurationError("workload outside host profile context scope")

    def cost(self, name, tokens, fallback):
        entry = self.document["costs"].get(name)
        return (
            fallback
            if entry is None
            else entry["fixed_us"] + tokens * entry["per_token_us"]
        )

    def describe(self):
        return dict(
            profile_id=self.document["profile_id"],
            fingerprint=self.fingerprint,
            timing_scope=self.document["timing_scope"],
            evidence_kind=self.document["evidence"]["kind"],
            covered_costs=tuple(self.document["costs"]),
            status="host_profile_loaded_not_full_engine_validation",
        )
