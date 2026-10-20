"""Require explicit approximation metadata for sampled native DAG execution."""

from scripts.lpx.validation_checks import require


def validate_replay_profiles(row):
    description = row.get("replay_profiles")
    require(isinstance(description, dict), "missing replay provenance")
    require(
        description.get("method") == "sampled_native_operator_dags", "replay method"
    )
    profiles = description.get("profiles")
    require(bool(profiles), "empty replay source profiles")
    fingerprints = set()
    for profile in profiles:
        require(profile.get("native_events", 0) > 0, "empty native profile")
        require(profile.get("native_duration_s", 0) > 0, "native profile duration")
        for key in ("input_sha256", "source_sha256", "node_body_sha256"):
            value = profile.get(key, "")
            require(
                isinstance(value, str)
                and len(value) == 64
                and all(c in "0123456789abcdef" for c in value),
                "invalid replay fingerprint",
            )
        fingerprints.add(profile["source_sha256"])
    require(len(fingerprints) == 1, "mixed replay source revisions")
    require(
        "token_events" in row and "decode_intervals" in row,
        "missing replay accounting events",
    )
