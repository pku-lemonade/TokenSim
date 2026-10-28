"""Extract per-provider output speeds from saved Artificial Analysis pages.

Reads the HTML of AA's provider pages for one model, saved for the default
10K workload and for the 100K workload, and writes each provider's median,
quartiles and 5th/95th percentiles of output speed for both workloads,
together with the per-workload medians the page embeds.  See
docs/lpx-aa-noise-check.md.
"""

import argparse
import json
from pathlib import Path
import re

ENTRY = re.compile(
    r'\{"id":"[0-9a-f-]+","slug":"[a-z0-9_.-]+","host":\{"name":"([^"]+)"'
)
SPEED = re.compile(
    r'"outputSpeed":\{"median":([\d.]+),"percentile05":([\d.]+),'
    r'"percentile95":([\d.]+),"quartile25":([\d.]+),"quartile75":([\d.]+)\}'
)
BY_PROMPT = re.compile(r'"(medium|long|hundredK)":\{"medianOutputSpeed":([\d.]+)')


def providers(path):
    """Output-speed statistics of each provider on one saved page."""
    text = path.read_text(encoding="utf-8", errors="replace").replace('\\"', '"')
    starts = [(m.start(), m.group(1)) for m in ENTRY.finditer(text)]
    result = {}
    for index, (start, name) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        segment = text[start:end]
        speed = SPEED.search(segment)
        if speed is None or name in result:
            continue
        values = map(float, speed.groups())
        result[name] = dict(
            zip(("median", "p05", "p95", "q25", "q75"), values),
            by_prompt={k: float(v) for k, v in BY_PROMPT.findall(segment)},
        )
    return result


def spread(stats):
    return dict(
        iqr=(stats["q75"] - stats["q25"]) / stats["median"],
        p05_p95=(stats["p95"] - stats["p05"]) / stats["median"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-10k", type=Path, required=True)
    parser.add_argument("--page-100k", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    short, long = providers(args.page_10k), providers(args.page_100k)
    rows = []
    for name, stats in short.items():
        row = dict(provider=name, workload_10k={**stats, **spread(stats)})
        if name in long:
            row["workload_100k"] = {**long[name], **spread(long[name])}
            row["ratio_100k_10k"] = long[name]["median"] / stats["median"]
        rows.append(row)
        print(
            f"{name:20s} 1k {stats['by_prompt'].get('medium', 0):6.1f}"
            f" 10k {stats['median']:6.1f}"
            f" 100k {long[name]['median'] if name in long else 0:6.1f}"
            f" IQR/median {row['workload_10k']['iqr']:.1%}"
        )
    args.summary.write_text(json.dumps(dict(source=args.source, rows=rows), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
