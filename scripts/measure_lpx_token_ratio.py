"""Measure Gemma 4 tokens per AA ``o200k_base`` token on public text.

Artificial Analysis counts output tokens with ``o200k_base``, while TokenSim
counts the model's native tokens.  This script tokenizes the same text with
both tokenizers and reports the ratio per text category.  It needs the
``tokenizers`` and ``tiktoken`` packages and a local copy of Gemma 4's
``tokenizer.json``.  See docs/lpx-token-ratio.md for the corpus and results.

Each ``--text`` argument is ``category=path``.  Plain files are read as text;
``.json`` files are read as a Wikipedia extract query or as Hugging Face
dataset-server rows with ``conversations``; Project Gutenberg files are cut to
the book between the START and END markers.  A ``#thought`` or ``#solution``
suffix on a dataset-server path keeps only that part of each OpenThoughts
answer, and any other suffix names the row field to read, such as
``#thinking_content`` or ``#final_answer``.
"""

import argparse
import hashlib
import json
from pathlib import Path


SECTIONS = dict(
    thought=("<|begin_of_thought|>", "<|end_of_thought|>"),
    solution=("<|begin_of_solution|>", "<|end_of_solution|>"),
)


def section(text, name):
    if name is None:
        return text
    begin, end = SECTIONS[name]
    start, stop = text.find(begin), text.find(end)
    return text[start + len(begin) : stop] if 0 <= start < stop else ""


def texts(path, part=None):
    raw = path.read_text(encoding="utf-8")
    if path.suffix != ".json":
        start, end = raw.find("*** START OF"), raw.find("*** END OF")
        if start >= 0 and end > start:
            raw = raw[raw.index("\n", start) + 1 : end]
        return [raw]
    data = json.loads(raw)
    if "query" in data:
        return [page.get("extract", "") for page in data["query"]["pages"].values()]
    rows = [row["row"] for row in data["rows"]]
    if part and part not in SECTIONS:
        return [row[part] for row in rows if row.get(part)]
    return [
        section(turn["value"], part)
        for row in rows
        for turn in row["conversations"]
        if turn["from"] == "assistant"
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-tokenizer", type=Path, required=True)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)

    import tiktoken
    from tokenizers import Tokenizer

    gemma = Tokenizer.from_file(str(args.gemma_tokenizer))
    o200k = tiktoken.get_encoding("o200k_base")
    totals, files = {}, []
    for item in args.text:
        category, spec = item.split("=", 1)
        path, _, part = spec.partition("#")
        path = Path(path)
        native = aa = characters = 0
        for text in texts(path, part or None):
            native += len(gemma.encode(text, add_special_tokens=False).ids)
            aa += len(o200k.encode(text, disallowed_special=()))
            characters += len(text)
        files.append(
            dict(
                category=category,
                file=path.name,
                part=part or None,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                characters=characters,
                gemma_tokens=native,
                o200k_tokens=aa,
                gemma_per_o200k=native / aa,
            )
        )
        total = totals.setdefault(category, dict(gemma_tokens=0, o200k_tokens=0))
        total["gemma_tokens"] += native
        total["o200k_tokens"] += aa
    for category, total in totals.items():
        total["gemma_per_o200k"] = total["gemma_tokens"] / total["o200k_tokens"]
        print(
            f"{category}: {total['gemma_tokens']} Gemma, {total['o200k_tokens']}"
            f" o200k, ratio {total['gemma_per_o200k']:.4f}"
        )
    if args.summary:
        args.summary.write_text(
            json.dumps(
                dict(
                    gemma_tokenizer_sha256=hashlib.sha256(
                        args.gemma_tokenizer.read_bytes()
                    ).hexdigest(),
                    tiktoken_encoding="o200k_base",
                    categories=totals,
                    files=files,
                ),
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
