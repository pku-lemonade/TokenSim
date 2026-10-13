# Contributing

Read this guide before making changes and follow the requirements that apply to your work.

## Development setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Code standards

- Keep every source, test and script file at 500 physical lines or fewer, including comments and blank lines.
- Split larger files into modules with clear responsibilities and keep their code readable.
- Before implementing a change, search the codebase for similar functionality, helpers, configuration fields and tests.
- Reuse or extend an existing implementation when it meets the requirement, and avoid duplicate logic and redundant abstractions.
- Keep configuration and validation, simulation behavior, latency estimation and result export in their respective modules.
- Use descriptive names, explicit inputs and small functions with a clear purpose, following the surrounding code's conventions.
- Validate inputs at module boundaries and report invalid configurations through the existing error types.
- Preserve deterministic behavior with explicit seeds and local random number streams.
- Keep changes focused on the requested behavior and avoid unrelated refactoring.
- Add or update meaningful tests for behavior changes, including relevant boundary cases and failure paths.
- Document changes to command-line arguments, configuration fields and result semantics.

## Checks

Format Python code with the project's pinned Black version before every commit.
The formatting step rewrites files, while the check confirms that formatting is complete.

```bash
python -m black TokenSim tests benchmark.py scripts examples/storage/hbf
python -m black --check TokenSim tests benchmark.py scripts examples/storage/hbf
python -m unittest discover \
  -s tests \
  -t .
git diff --check
```

Run the formatter, tests and diff check before committing or submitting a code change.
`python -m pytest` runs the same tests.
Tests use small deterministic inputs and do not assert historical throughput values.
For documentation-only changes, check sentence layout, prohibited expressions, links and any edited examples.
Review the diff to confirm that it contains the intended changes and preserves unrelated work.

## Changes that affect simulated results

Simulations are deterministic for a given input and seed.
For changes intended to preserve results, such as refactoring or performance work, compare the result JSON and, for AgentX replays, the `replay/*.jsonl` logs of equivalent runs.
These outputs should match except for fields reporting the time taken to execute the simulation.
For changes intended to alter results, describe the expected effect and measured difference in the pull request.

## Operator data

Operator packages live in `data/operator_data/<device>/<backend>`.
Import AIConfigurator measurements with the operator-data CLI as described in [docs/aiconfigurator.md](docs/aiconfigurator.md), or collect your own with [scripts/collect](scripts/collect/README.md).
Keep each package's `generation_meta.yaml` because it records the source, version and operations included in each measurement.
Validate a package before committing it.

```bash
tokensim-operator-data validate data/operator_data/h100_sxm/vllm
```

## Documentation standards

These requirements apply to all project documentation, including Markdown files, configuration examples and contributor guidance.
Write project guidance in English, and preserve the established language and section structure of other documents unless a change is requested.

- Put each complete prose sentence on its own line and do not wrap a sentence to meet a column limit.
- Use headings, tables, formulas and code blocks in their appropriate formats.
- Put each configuration argument or field on its own line, and use `\` to continue multi-line shell commands.
- Describe behavior, conditions and results directly with specific subjects and verbs.
- Expand abbreviations at their first use and explain what technical terms refer to in the current context.
- Split long sentences into complete sentences that each express a clear idea.
- Preserve technical meaning, configuration values, numerical data, formulas, links and anchors when polishing text.
- Support factual corrections with implementation code or original data.
- Check sentence layout, prohibited expressions and `git diff --check` after editing.
- Check shell syntax and compare JSON/YAML values when changing examples.

### Prohibited wording

Do not use the following expressions or equivalent wording as narrative framing, promotional claims, transitions or conclusions.
Apply these restrictions to equivalent expressions in any language.
Quoted catalog entries document the rules, and code identifiers or necessary technical descriptions must retain their precise meanings.

| Category | Prohibited expressions |
| --- | --- |
| Claims of objectivity or sincerity | "no hype, no hate"; "neither bragging nor flattery"; "let's call it as it is"; "sticking to the facts"; "objectively speaking"; "in all fairness"; "to be honest"; "to give a fair assessment"; "I'm not bragging"; "honestly"; "don't scroll away"; "read this carefully to the end"; "this really isn't an ad" |
| Promises of complete or effortless understanding | "master it inside out"; "explain every aspect"; "make it completely clear"; "understand it completely"; "one article explains everything"; "one explanation covers everything"; "fully figure it out"; "never again"; "understand instantly"; "understand it in one article"; "this is all you need to read"; "the most comprehensive"; "cover absolutely everything"; "see through it all"; "get it all figured out"; "hold your hand through it"; "let me say this first" |
| Emotional labels without concrete information | "grounded"; "quiet confidence from being grounded"; "standing on solid ground"; "I only understood later"; "the sooner you understand, the better"; "underlying logic"; "cognitive awakening"; "broadening your mindset"; "truly capable people"; "set yourself apart"; "packed with actionable insights"; "from the bottom of my heart"; "my sincere advice"; "pure actionable insights" |
| Canned transitions and endings | "first"; "second"; "finally"; "next up"; "key takeaway"; "highlight this"; "to summarize"; "at the end of the day"; "all in all"; "it is worth mentioning"; "now here is the question"; "have you ever wondered"; "to put it bluntly"; "did you know"; "it has to be said"; "no gimmicks"; "only sharing what's most real" |
| Filler and formulaic comparisons | "not only ... but also ..."; "rather than X, choose Y"; "actually"; "in fact"; "to some extent"; "in a sense"; "it should be noted"; "it is worth emphasizing" |
| Business and self-improvement jargon | "empower"; "break through an impasse"; "underlying logic"; "leap to a new level"; "elevate"; "transform"; "dimensionality-reduction attack"; "overtake on the curve"; "upgrade your cognition"; "higher-dimensional perspective"; "close the loop"; "actionable lever"; "end-to-end chain"; "ecological niche"; "sense of value" |

### Prohibited sentence patterns

Do not use the following sentence patterns or close variants.

| Prohibited pattern | Direct wording |
| --- | --- |
| "Not X, but Y." | Describe the required behavior directly. |
| "Before X; now Y." | State the current behavior and give necessary historical facts without a stock comparison. |
| "Remember, what really matters is ..." | State the requirement without a lecturing introduction. |
| "It is worth noting that ..." | State the fact directly. |
| "In short", "in conclusion" or "at the end of the day" as a closing. | End when the useful information is complete. |
| "In this age of X, every one of us ..." | Describe the specific system or situation. |
| Any sentence beginning with "Let us" or "Let's". | Name the action or responsible component. |
| "Not only X, but also Y." | List the relevant properties directly. |
| "Rather than X, choose Y." | State the recommended action and its reason. |

### Terms requiring a specific reason

Use "depth", "involution", "resonance", "disruption", "race track" or "competitive arena", "moat", "postmortem review", "accumulation", "deliberate practice" and "growth mindset" only when their precise meaning is clear and the context requires them.
Avoid using them as vague emotional labels or promotional metaphors.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
