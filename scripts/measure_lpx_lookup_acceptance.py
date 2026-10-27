"""Replay prompt-lookup drafting over recorded Gemma 4 31B outputs.

The outputs were generated at temperature 0, so a draft token is accepted
exactly when it equals the recorded token.  The script needs the
``tokenizers`` package and Gemma 4's ``tokenizer.json``.  Each ``--data``
argument is ``name=path`` for Hugging Face dataset-server rows with ``input``
and ``raw_output`` fields, or ``name=path#part`` for rows with a user and an
assistant message whose ``reasoning`` field holds the thinking; ``part`` is
one of MESSAGE_PARTS.  See docs/lpx-lookup-acceptance.md and
docs/lpx-lookup-coding.md.
"""

import argparse
import hashlib
import json
from pathlib import Path

DRAFT_TOKENS = (1, 3, 5, 7)
MINIMUM_NGRAMS = (1, 2)
MAXIMUM_NGRAM = 3
PROMPTS = ("full", "half", "none")
# Prompt and output texts of a user/assistant row: the thinking followed by
# the answer, the thinking alone, the answer after the thinking, which the
# drafter can copy from, and the answer without the thinking.
MESSAGE_PARTS = {
    "combined": lambda q, r, a: (q, r + "\n\n" + a),
    "thinking": lambda q, r, a: (q, r),
    "answer": lambda q, r, a: (q + "\n\n" + r, a),
    "answer-alone": lambda q, r, a: (q, a),
}


def texts(row, part=None):
    """Prompt and output text of one dataset-server row."""
    if part is None:
        return row["input"], row["raw_output"]
    user, assistant = row["messages"]
    return MESSAGE_PARTS[part](
        user["content"], assistant.get("reasoning") or "", assistant["content"]
    )


def equivalent_acceptance(tau, draft_tokens):
    """Per-position acceptance giving ``tau`` committed tokens with K proposals."""
    if tau <= 1:
        return 0.0
    if tau >= draft_tokens + 1:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(100):
        alpha = (low + high) / 2
        if (1 - alpha ** (draft_tokens + 1)) / (1 - alpha) < tau:
            low = alpha
        else:
            high = alpha
    return (low + high) / 2


def replay(prompt, output, draft_tokens, minimum):
    """Steps, proposing steps, accepted draft tokens and proposal lengths."""
    sequence = prompt + output
    index = {}

    def insert(end):
        for n in range(minimum, MAXIMUM_NGRAM + 1):
            if end >= n:
                index[tuple(sequence[end - n : end])] = end

    for end in range(1, len(prompt)):
        insert(end)
    position, steps, proposing, accepted = len(prompt), 0, 0, 0
    lengths = [0] * (draft_tokens + 1)
    while position < len(sequence):
        proposal = []
        for n in range(MAXIMUM_NGRAM, minimum - 1, -1):
            if position >= n:
                start = index.get(tuple(sequence[position - n : position]))
                if start is not None:
                    proposal = sequence[start : min(start + draft_tokens, position)]
                    break
        matched = 0
        for offset, token in enumerate(proposal):
            if (
                position + offset >= len(sequence)
                or sequence[position + offset] != token
            ):
                break
            matched += 1
        commit = min(matched + 1, len(sequence) - position)
        steps += 1
        proposing += bool(proposal)
        lengths[len(proposal)] += 1
        accepted += min(matched, commit)
        for end in range(position, position + commit):
            insert(end)
        position += commit
    return steps, proposing, accepted, lengths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-tokenizer", type=Path, required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.gemma_tokenizer))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    results, files = [], []
    for item in args.data:
        name, path = item.split("=", 1)
        path, _, part = path.partition("#")
        path = Path(path)
        rows = [row["row"] for row in json.loads(path.read_text())["rows"]]
        pairs = [
            tuple(encode(text) for text in texts(row, part or None)) for row in rows
        ]
        files.append(
            dict(
                name=name,
                file=path.name,
                part=part or None,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=len(rows),
                prompt_tokens=sum(len(p) for p, _ in pairs),
                output_tokens=sum(len(o) for _, o in pairs),
            )
        )
        for prompt_mode in PROMPTS:
            for minimum in MINIMUM_NGRAMS:
                for draft_tokens in DRAFT_TOKENS:
                    steps = proposing = accepted = tokens = 0
                    lengths = [0] * (draft_tokens + 1)
                    for prompt, output in pairs:
                        if prompt_mode == "half":
                            prompt = prompt[: len(prompt) // 2]
                        elif prompt_mode == "none":
                            prompt = []
                        s, p, a, counts = replay(prompt, output, draft_tokens, minimum)
                        lengths = [x + y for x, y in zip(lengths, counts)]
                        steps, proposing, accepted = (
                            steps + s,
                            proposing + p,
                            accepted + a,
                        )
                        tokens += len(output)
                    tau = tokens / steps
                    row = dict(
                        data=name,
                        prompt=prompt_mode,
                        minimum_ngram=minimum,
                        draft_tokens=draft_tokens,
                        steps=steps,
                        tau=tau,
                        proposing_share=proposing / steps,
                        accepted_per_proposal=accepted / max(proposing, 1),
                        equivalent_acceptance=equivalent_acceptance(tau, draft_tokens),
                        proposal_length_shares=[x / steps for x in lengths],
                    )
                    results.append(row)
                    print(
                        f"{name} prompt={prompt_mode} n_min={minimum} K={draft_tokens}:"
                        f" tau {tau:.3f}, proposing {row['proposing_share']:.2f},"
                        f" alpha {row['equivalent_acceptance']:.3f}",
                        flush=True,
                    )
    if args.summary:
        args.summary.write_text(
            json.dumps(
                dict(
                    gemma_tokenizer_sha256=hashlib.sha256(
                        args.gemma_tokenizer.read_bytes()
                    ).hexdigest(),
                    maximum_ngram=MAXIMUM_NGRAM,
                    files=files,
                    results=results,
                ),
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
