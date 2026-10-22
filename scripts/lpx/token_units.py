"""Separate native model tokens from AA o200k_base accounting units."""

from dataclasses import asdict, dataclass
import hashlib
import math

from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class TokenUnits:
    native_per_aa_input: float = 1.0
    native_per_aa_output: float = 1.0
    source: str = "Unmeasured identity assumption."
    measured: bool = False
    corpus_sha256: str | None = None
    tokenizer: str | None = None
    revision: str | None = None

    def __post_init__(self):
        for name in ("native_per_aa_input", "native_per_aa_output"):
            number(getattr(self, name), name, positive=True)
        if (
            type(self.measured) is not bool
            or not isinstance(self.source, str)
            or not self.source
        ):
            raise ConfigurationError(
                "token-unit provenance requires a source and measured boolean"
            )
        if self.measured and (
            not self.corpus_sha256 or not self.tokenizer or not self.revision
        ):
            raise ConfigurationError(
                "measured token ratios require corpus hash, tokenizer and revision"
            )

    def native_lengths(self, aa_input, aa_output):
        integer(aa_input, "AA input tokens", minimum=1)
        integer(aa_output, "AA output tokens", minimum=2)
        # The timing interval excludes one native output, not one arbitrary AA token.
        return max(1, math.ceil(aa_input * self.native_per_aa_input)), max(
            2, math.ceil(aa_output * self.native_per_aa_output)
        )

    def aa_intervals(self, native_output):
        integer(native_output, "native output tokens", minimum=2)
        return (native_output - 1) / self.native_per_aa_output

    def report(self):
        return dict(
            **asdict(self),
            comparison_status=(
                "Corpus-measured token conversion; endpoint text may differ."
                if self.measured
                else "AA comparison is conditional on unmeasured token ratios."
            ),
        )


def measure(samples, native_encode, aa_encode, *, tokenizer, revision):
    """Count the same text with both tokenizers; callbacks simplify offline use."""
    rows, totals, digest = (
        [],
        {kind: [0, 0] for kind in ("input", "output")},
        hashlib.sha256(),
    )
    for index, sample in enumerate(samples):
        if set(sample) != {"input", "output"} or any(
            not isinstance(v, str) or not v for v in sample.values()
        ):
            raise ConfigurationError(
                "token samples require nonempty input and output text"
            )
        record = dict(index=index)
        for kind in ("input", "output"):
            text = sample[kind]
            digest.update(
                kind.encode()
                + b"\0"
                + len(text.encode()).to_bytes(8, "big")
                + text.encode()
            )
            native, aa = len(native_encode(text)), len(aa_encode(text))
            if not native or not aa:
                raise ConfigurationError("token sample encoders must produce tokens")
            totals[kind][0] += native
            totals[kind][1] += aa
            record[kind] = dict(native=native, aa=aa, native_per_aa=native / aa)
        rows.append(record)
    if not rows:
        raise ConfigurationError("token conversion requires a nonempty corpus")
    units = TokenUnits(
        totals["input"][0] / totals["input"][1],
        totals["output"][0] / totals["output"][1],
        "Same-text native tokenizer versus tiktoken o200k_base; special tokens disabled.",
        True,
        digest.hexdigest(),
        tokenizer,
        revision,
    )
    return dict(
        token_units=asdict(units),
        samples=rows,
        totals=totals,
        estimator="Ratio of corpus totals, separated by input/output.",
    )
