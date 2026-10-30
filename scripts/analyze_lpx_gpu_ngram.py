"""Compare measured n-gram turns with the existing lookup replay by identity.

Export the measured turns outside the repository with the GPU client first.
The replay retains its existing text tokenization and most-recent-match rule;
the comparison does not attribute every difference to the matching rule.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import WorkloadValidationError
from scripts.analyze_lpx_gpu_smoke import read_outputs
from scripts.lpx.gpu_metrics import ngram_acceptance, output_quality
from scripts.measure_lpx_lookup_acceptance import equivalent_acceptance, replay, texts
from scripts.search_lpx_native import write_json


def validate_server_command(command):
    argv = command.get("argv") if isinstance(command, dict) else None
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
        raise WorkloadValidationError("Recorded server command needs string arguments")
    try:
        speculative = json.loads(argv[argv.index("--speculative-config") + 1])
        sequences = int(argv[argv.index("--max-num-seqs") + 1])
    except (ValueError, IndexError, TypeError) as error:
        raise WorkloadValidationError(
            "Missing recorded n-gram server settings"
        ) from error
    expected = dict(
        method="ngram",
        num_speculative_tokens=7,
        prompt_lookup_min=1,
        prompt_lookup_max=3,
    )
    if (
        not isinstance(speculative, dict)
        or sequences != 1
        or any(speculative.get(key) != value for key, value in expected.items())
    ):
        raise WorkloadValidationError("Recorded server does not match T4 settings")
    runner = command.get("environment", {}).get("VLLM_USE_V2_MODEL_RUNNER")
    return {"0": "v1-cpu", "1": "v2-gpu"}.get(runner, "unknown")


def matching_rules(implementation):
    if implementation == "v1-cpu":
        return [
            "vLLM V1 CPU selects the earliest match; replay selects the most recent match"
        ]
    if implementation == "v2-gpu":
        return [
            "vLLM V2 GPU and replay select the longest n-gram and then the most recent match",
            "vLLM V2 GPU fills missing draft slots with the last sampled token; replay uses variable proposal lengths",
        ]
    if implementation == "unknown":
        return [
            "recorded runner setting does not identify the native matching implementation"
        ]
    raise WorkloadValidationError("Unknown n-gram implementation")


def compare_turn(turn, exported, tokenizer, draft_length):
    messages = exported["messages"]
    if (
        len(messages) != 2
        or messages[1]["content"] != turn["content"]
        or (messages[1].get("reasoning") or "") != turn["reasoning"]
    ):
        raise WorkloadValidationError("Exported text differs from the measured turn")
    prompt_text, output_text = texts(exported, "combined")
    prompt = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    output = tokenizer.encode(output_text, add_special_tokens=False).ids
    if not output:
        raise WorkloadValidationError("Lookup comparison needs nonempty replay tokens")
    steps, proposing, accepted, counts = replay(prompt, output, draft_length, 1)
    measured = ngram_acceptance(turn, draft_length)
    replay_tau = len(output) / steps
    replay_alpha = equivalent_acceptance(replay_tau, draft_length)
    return dict(
        native=measured,
        replay=dict(
            prompt_tokens=len(prompt),
            output_tokens=len(output),
            steps=steps,
            proposing_steps=proposing,
            accepted_tokens=accepted,
            proposal_length_counts=counts,
            tau=replay_tau,
            equivalent_acceptance=replay_alpha,
        ),
        native_prompt_tokens=turn["prompt_tokens"],
        tau_difference=measured["tau"] - replay_tau,
        equivalent_acceptance_difference=measured["equivalent_acceptance"]
        - replay_alpha,
        quality=output_quality(turn),
    )


def compare(outputs, exported_rows, tokenizer, implementation="unknown"):
    rules = matching_rules(implementation)
    configurations = {json.dumps(row["config"], sort_keys=True) for row in outputs}
    if len(configurations) != 1:
        raise WorkloadValidationError("Compare one nonempty n-gram configuration")
    config = outputs[0]["config"]
    if config["method"] != "ngram" or config["K"] != 7:
        raise WorkloadValidationError("T4 comparison requires n-gram K=7")
    measured = {}
    for request in outputs:
        for turn in request["turns"]:
            key = (request["id"], turn["turn"])
            if key in measured:
                raise WorkloadValidationError("Duplicate measured turn identity")
            measured[key] = (request, turn)
    exported = {}
    for row in exported_rows:
        key = (row["id"], row["turn"])
        if key in exported:
            raise WorkloadValidationError("Duplicate exported turn identity")
        exported[key] = row
    if not measured or measured.keys() != exported.keys():
        raise WorkloadValidationError("Measured and exported turn identities differ")
    results = []
    for key, (request, turn) in sorted(measured.items()):
        if exported[key]["meta"] != request["meta"]:
            raise WorkloadValidationError("Exported request metadata differs")
        results.append(
            dict(
                id=key[0],
                turn=key[1],
                set=request["set"],
                meta=request["meta"],
                **compare_turn(turn, exported[key], tokenizer, config["K"]),
            )
        )
    groups = []
    for dataset in sorted({row["set"] for row in results}):
        selected = [row for row in results if row["set"] == dataset]
        native_tau = statistics.mean(row["native"]["tau"] for row in selected)
        replay_tau = statistics.mean(row["replay"]["tau"] for row in selected)
        native_alpha = equivalent_acceptance(native_tau, config["K"])
        replay_alpha = equivalent_acceptance(replay_tau, config["K"])
        difference = native_alpha - replay_alpha
        groups.append(
            dict(
                set=dataset,
                turn_count=len(selected),
                native_mean_tau=native_tau,
                replay_mean_tau=replay_tau,
                mean_tau_difference=native_tau - replay_tau,
                native_equivalent_acceptance=native_alpha,
                replay_equivalent_acceptance=replay_alpha,
                equivalent_acceptance_difference=difference,
                mean_absolute_turn_acceptance_difference=statistics.mean(
                    abs(row["equivalent_acceptance_difference"]) for row in selected
                ),
                expectation_within_0_03=abs(difference) < 0.03,
            )
        )
    return dict(
        config=config,
        minimum_ngram=1,
        maximum_ngram=3,
        native_implementation=implementation,
        differences=[
            *rules,
            "replay tokenizes exported text without the server chat-template tokens",
            "combined replay inserts two newlines between reasoning and answer",
            "native tau excludes the prefill token; replay counts all replay output tokens",
            "native output includes generated control tokens hidden by the text parser",
            "multi-turn exports flatten conversation history into user text",
        ],
        sets=groups,
        turns=results,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--gemma-tokenizer", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--server-command", type=Path, required=True)
    parser.add_argument(
        "--validated-engine-steps",
        action="store_true",
        required=True,
        help="Confirm source and plain/MTP counter checks at concurrency one",
    )
    args = parser.parse_args(argv)
    server_command = args.server_command.read_bytes()
    implementation = validate_server_command(json.loads(server_command))
    from tokenizers import Tokenizer

    files, rows = [], []
    for path in sorted(args.rows.glob("*.json")):
        content = path.read_bytes()
        rows.extend(record["row"] for record in json.loads(content)["rows"])
        files.append(dict(file=path.name, sha256=hashlib.sha256(content).hexdigest()))
    result = compare(
        read_outputs(args.outputs),
        rows,
        Tokenizer.from_file(str(args.gemma_tokenizer)),
        implementation=implementation,
    )
    result["export_files"] = files
    result["server_command_sha256"] = hashlib.sha256(server_command).hexdigest()
    result["tokenizer_sha256"] = hashlib.sha256(
        args.gemma_tokenizer.read_bytes()
    ).hexdigest()
    write_json(args.summary, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
