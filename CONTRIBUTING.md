# Contributing

## Development setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Python 3.11 is also supported.

## Checks

Run the test suite and the formatter before sending a change:

```bash
python -m unittest discover -s tests -t .
black --check TokenSim tests benchmark.py scripts
```

`python -m pytest` runs the same tests. Tests use small deterministic inputs;
they do not assert historical throughput values.

## Changes that affect simulated results

Simulations are deterministic for a given input and seed. When a change is
meant to keep results unchanged (refactoring, performance work), compare the
result JSON and, for AgentX replays, the `replay/*.jsonl` logs of a run before
and after the change; they should match except for wall-clock fields. When a
change is meant to alter results, describe the expected effect and the measured
difference in the pull request.

## Operator data

Operator packages live in `data/operator_data/<device>/<backend>`. Import
AIConfigurator measurements with the operator-data CLI as described in
[docs/aiconfigurator.md](docs/aiconfigurator.md), or collect your own with
[scripts/collect](scripts/collect/README.md). Keep each package's
`generation_meta.yaml`: it records the source, version and timing scope of
every row. Validate a package before committing it:

```bash
tokensim-operator-data validate data/operator_data/<device>/<backend>
```

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0 (see [LICENSE](LICENSE)).
