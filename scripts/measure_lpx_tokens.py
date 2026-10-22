"""Measure native/AA token ratios on a JSONL corpus with input/output text."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import ConfigurationError
from scripts.lpx.token_units import measure
from scripts.search_lpx_native import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        import tiktoken
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.revision, trust_remote_code=False
        )
        aa = tiktoken.get_encoding("o200k_base")
        samples = [
            json.loads(line)
            for line in args.samples.read_text().splitlines()
            if line.strip()
        ]
        result = measure(
            samples,
            lambda text: tokenizer.encode(text, add_special_tokens=False),
            lambda text: aa.encode(text, disallowed_special=()),
            tokenizer=args.tokenizer,
            revision=args.revision,
        )
        result["packages"] = {
            "tiktoken": tiktoken.__version__,
            "transformers": __import__("transformers").__version__,
        }
        write_json(args.output, result)
        print(json.dumps(result["token_units"], indent=2))
    except (ImportError, OSError, ValueError, ConfigurationError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
