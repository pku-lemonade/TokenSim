#!/usr/bin/env python3
"""Build a verified conversation-to-DP-rank mapping from reference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PATTERN = re.compile(
    r"Hash key 'header:x-session-id:([^']+)' mapped to worker: \S+@(\d+)"
)


def build(router_log: Path, profile_export: Path) -> dict:
    key_to_rank = {}
    for match in PATTERN.finditer(router_log.read_text(errors="replace")):
        key_to_rank[match.group(1)] = int(match.group(2))
    metadata = [
        json.loads(line)["metadata"]
        for line in profile_export.read_text().splitlines()
        if line.strip()
    ]
    conversations = {}
    for item in metadata:
        conversation = item.get("conversation_id")
        correlation = item.get("x_correlation_id")
        if conversation and correlation:
            conversations[conversation] = correlation
    mapping = {
        conversation: key_to_rank[correlation]
        for conversation, correlation in conversations.items()
        if correlation in key_to_rank
    }
    return {
        "schema_version": 1,
        "mapping": mapping,
        "coverage": {
            "conversation_count": len(conversations),
            "mapped_conversation_count": len(mapping),
            "coverage": len(mapping) / len(conversations) if conversations else 0.0,
            "unmapped_examples": [
                conversation
                for conversation in conversations
                if conversation not in mapping
            ][:20],
        },
        "provenance": {
            "router_log": str(router_log),
            "router_log_sha256": hashlib.sha256(router_log.read_bytes()).hexdigest(),
            "profile_export": str(profile_export),
            "profile_export_sha256": hashlib.sha256(profile_export.read_bytes()).hexdigest(),
            "key_field": "metadata.x_correlation_id -> router header:x-session-id",
            "worker_field": "router log @rank suffix",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-log", type=Path, required=True)
    parser.add_argument("--profile-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.router_log, args.profile_export)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["coverage"], indent=2))
