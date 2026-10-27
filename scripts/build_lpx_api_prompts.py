"""Build the Gemma 4 31B API request sets of the LPX workload study.

Writes one JSON line per request with ``id``, ``set``, ``turns`` (user
messages sent in order), ``reasoning``, ``max_tokens`` and ``meta``:

- ``speed-coding``: the 80 coding prompts of SPEED-Bench's qualitative split,
  each with thinking on and off.
- ``speed-cc16k``: code-completion prompts of SPEED-Bench's 16K throughput
  split, chosen with a fixed seed, with thinking off.
- ``aa-docs``: Project Gutenberg books cut to 10K and 100K Gemma tokens with a
  summary question and a quotation question, with thinking on.

Needs ``pyarrow`` and ``tokenizers``.  See docs/lpx-api-outputs.md.
"""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.measure_lpx_token_ratio import texts

CONTEXTS = (10000, 100000)
QUESTIONS = dict(
    summary=(
        "Write a detailed report on the text above of at least 2,000 words."
        " Summarize its content in order, then explain its main themes or"
        " arguments."
    ),
    quotes=(
        "Using only the text above, answer in at least 2,000 words: what are"
        " the five most important moments or arguments in this text? For each,"
        " quote the relevant passages directly and explain their significance."
    ),
)
# Room for chat-template tokens around the user message.
TEMPLATE_TOKENS = 16
COMPLETIONS = 40
SEED = 20261027


def read_parquet(path):
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def speed_coding(path):
    for row in read_parquet(path):
        if row["category"] != "coding":
            continue
        for reasoning in (True, False):
            yield dict(
                id=f"speed-coding/{row['question_id']}/{'think' if reasoning else 'plain'}",
                set="speed-coding",
                turns=list(row["turns"]),
                reasoning=reasoning,
                max_tokens=16384,
                meta=dict(
                    question_id=row["question_id"],
                    source=row["source"],
                    src_id=row["src_id"],
                ),
            )


def speed_completion(path, count):
    rows = [r for r in read_parquet(path) if r["sub_category"] == "code_completion"]
    rows.sort(key=lambda r: r["question_id"])
    for row in random.Random(SEED).sample(rows, count):
        yield dict(
            id=f"speed-cc16k/{row['question_id']}",
            set="speed-cc16k",
            turns=list(row["turns"]),
            reasoning=False,
            max_tokens=4096,
            meta=dict(
                question_id=row["question_id"],
                source=row["source"],
                src_id=row["src_id"],
            ),
        )


def aa_documents(directory, tokenizer):
    for path in sorted(directory.glob("pg*.txt")):
        book = texts(path)[0]
        title = next(
            (line[6:].strip() for line in path.open() if line.startswith("Title:")),
            path.stem,
        )
        ids = tokenizer.encode(book, add_special_tokens=False).ids
        for context in CONTEXTS:
            for style, question in QUESTIONS.items():
                question_tokens = len(
                    tokenizer.encode(question, add_special_tokens=False).ids
                )
                keep = context - question_tokens - TEMPLATE_TOKENS - 2
                document = tokenizer.decode(ids[:keep])
                yield dict(
                    id=f"aa-docs/{path.stem}/{context // 1000}k/{style}",
                    set="aa-docs",
                    turns=[document + "\n\n" + question],
                    reasoning=True,
                    max_tokens=16384,
                    meta=dict(
                        book=path.stem,
                        title=title,
                        context=context,
                        style=style,
                        document_tokens=keep,
                    ),
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-tokenizer", type=Path, required=True)
    parser.add_argument("--speed-qualitative", type=Path, required=True)
    parser.add_argument("--speed-16k", type=Path, required=True)
    parser.add_argument("--gutenberg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.gemma_tokenizer))
    requests = [
        *speed_coding(args.speed_qualitative),
        *speed_completion(args.speed_16k, COMPLETIONS),
        *aa_documents(args.gutenberg, tokenizer),
    ]
    with args.output.open("w") as handle:
        for request in requests:
            handle.write(json.dumps(request) + "\n")
    counts = {}
    for request in requests:
        counts[request["set"]] = counts.get(request["set"], 0) + 1
    print(f"{len(requests)} requests: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
