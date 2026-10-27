"""Replay prompt-lookup drafting over long multi-turn Gemma 4 31B contexts.

Each ``--data`` argument is a Hugging Face dataset-server rows file whose rows
hold a ``conversations`` list of ``role``/``content`` messages.  Assistant
turns are replayed with the drafter of measure_lpx_lookup_acceptance.py, and
user turns only extend the text available for copying.  A lookback window
limits copying to occurrences ending within the last ``W`` tokens.  The
script needs the ``tokenizers`` package and Gemma 4's ``tokenizer.json``.
See docs/lpx-lookup-context.md.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.measure_lpx_lookup_acceptance import MAXIMUM_NGRAM, equivalent_acceptance

DRAFT_TOKENS = (3, 7)
MINIMUM_NGRAM = 1
WINDOWS = (None, 16384, 4096, 1024)
BINS = (0, 2048, 4096, 8192, 16384, 32768)


def context_bin(context):
    """Lower edge of the context-length bin."""
    return max(edge for edge in BINS if context >= edge)


def replay_conversation(messages, draft_tokens, window):
    """Per assistant turn: context, tokens, steps, proposing steps, lengths."""
    sequence, index, turns = [], {}, []

    def insert(end):
        for n in range(MINIMUM_NGRAM, MAXIMUM_NGRAM + 1):
            if end >= n:
                index[tuple(sequence[end - n : end])] = end

    for role, tokens in messages:
        start = len(sequence)
        if role != "assistant":
            sequence.extend(tokens)
            for end in range(start, len(sequence)):
                insert(end)
            continue
        sequence.extend(tokens)
        position, steps, proposing = start, 0, 0
        lengths = [0] * (draft_tokens + 1)
        while position < len(sequence):
            proposal = []
            for n in range(MAXIMUM_NGRAM, MINIMUM_NGRAM - 1, -1):
                if position >= n:
                    found = index.get(tuple(sequence[position - n : position]))
                    # The index keeps the latest occurrence, so one outside the
                    # window means none inside it; a shorter suffix may match.
                    if found is not None and (
                        window is None or found >= position - window
                    ):
                        proposal = sequence[found : min(found + draft_tokens, position)]
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
            for end in range(position, position + commit):
                insert(end)
            position += commit
        turns.append((start, len(tokens), steps, proposing, lengths))
    return turns


def summarize(totals, draft_tokens):
    tokens, steps, proposing, lengths = totals
    tau = tokens / steps
    return dict(
        tokens=tokens,
        steps=steps,
        tau=tau,
        proposing_share=proposing / steps,
        equivalent_acceptance=equivalent_acceptance(tau, draft_tokens),
        proposal_length_shares=[x / steps for x in lengths],
    )


def measure(conversations, draft_tokens, window):
    groups = {}
    for messages in conversations:
        for context, tokens, steps, proposing, lengths in replay_conversation(
            messages, draft_tokens, window
        ):
            for key in ("all", context_bin(context)):
                total = groups.setdefault(key, [0, 0, 0, [0] * (draft_tokens + 1)])
                total[0] += tokens
                total[1] += steps
                total[2] += proposing
                total[3] = [x + y for x, y in zip(total[3], lengths)]
    return {str(key): summarize(total, draft_tokens) for key, total in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-tokenizer", type=Path, required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.gemma_tokenizer))
    encode = lambda text: tokenizer.encode(text or "", add_special_tokens=False).ids
    conversations, files = [], []
    for path in args.data:
        rows = [row["row"] for row in json.loads(path.read_text())["rows"]]
        conversations.extend(
            [(m["role"], encode(m["content"])) for m in row["conversations"]]
            for row in rows
        )
        files.append(
            dict(
                file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=len(rows),
            )
        )
    results = []
    for window in WINDOWS:
        for draft_tokens in DRAFT_TOKENS:
            groups = measure(conversations, draft_tokens, window)
            results.append(dict(window=window, draft_tokens=draft_tokens, **groups))
            print(
                f"W={window} K={draft_tokens}: "
                + ", ".join(
                    f"{key} {value['equivalent_acceptance']:.3f}"
                    for key, value in groups.items()
                ),
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
                    minimum_ngram=MINIMUM_NGRAM,
                    context_bins=BINS,
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
