"""Physical growth model for extrapolating complete DSV4 attention modules.

The FLOP/byte decomposition follows AIConfigurator's dsv4_attention_sol_ms,
with projection/compressor dtypes and cache entry sizes explicit. This is a
growth model anchored to measured latency, not an absolute kernel benchmark.
"""

from dataclasses import asdict, dataclass
import math

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import dtype_bytes


def causal_limited_pairs(batch, queries, prefix, limit):
    """Sum min(prefix + i + 1, limit) in constant time."""
    if queries <= 0 or limit <= 0:
        return 0
    ramp = min(queries, max(0, limit - prefix))
    return batch * (ramp * prefix + ramp * (ramp + 1) // 2 + (queries - ramp) * limit)


def _sum_floor(n, divisor):
    if n < 0:
        return 0
    quotient, remainder = divmod(n, divisor)
    return divisor * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


def compressed_context_pairs(batch, queries, prefix, ratio, limit):
    """Sum min(floor((prefix + i + 1) / ratio), limit)."""
    if queries <= 0 or ratio <= 0 or limit <= 0:
        return 0
    ramp_end = min(prefix + queries, limit * ratio - 1)
    ramp_queries = max(0, ramp_end - prefix)
    ramp = _sum_floor(ramp_end, ratio) - _sum_floor(prefix, ratio) if ramp_queries else 0
    return batch * (ramp + (queries - ramp_queries) * limit)


@dataclass(frozen=True)
class AttentionSOLConfig:
    # Borrowed FlashMLA table's layouts, inferred from the collector defaults
    # and read-only vLLM source; historical 0.24.0 layout is not independently
    # measured. No source-to-serving layout ratio is applied in this stage.
    kv_state_bytes: float = 584.0
    indexer_state_bytes: float = 132.0
    projection_dtype: str = "fp8_block"
    output_absorption_dtype: str = "fp8_block"
    compressor_dtype: str = "bf16"
    attention_dtype: str = "bf16"
    indexer_dtype: str = "fp8"

    def __post_init__(self):
        for name in ("kv_state_bytes", "indexer_state_bytes"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ConfigurationError(f"attention SOL {name} must be finite and positive")


@dataclass(frozen=True)
class AttentionSOLWork:
    flops: dict
    bytes: dict
    math_seconds: float
    memory_seconds: float

    @property
    def latency_us(self):
        return max(self.math_seconds, self.memory_seconds) * 1e6

    def to_dict(self):
        return {**asdict(self), "latency_us": self.latency_us}


class DeepSeekV4AttentionSOL:
    def __init__(self, model, device, config=None):
        if model.deepseek_v4 is None:
            raise ConfigurationError("attention SOL requires DSV4 model geometry")
        self.model = model
        self.device = device
        self.config = config or AttentionSOLConfig()
        self.compute_rates = {
            name: device.peak_compute_for(dtype).value
            for name, dtype in {
                "projection": self.config.projection_dtype,
                "output_absorption": self.config.output_absorption_dtype,
                "compressor": self.config.compressor_dtype,
                "indexer": self.config.indexer_dtype,
                "indexer_weights": "bf16",
                "attention": self.config.attention_dtype,
            }.items()
        }
        self.bandwidth = device.memory_bandwidth_bytes_per_s.value
        self._growth_geometry = {}
        self._shared_geometry = {}

    def limits_us(self, key):
        """Return math/memory bounds and indexer bytes without building a report.

        Cache only length-independent physical terms, not fitted latency. The
        diagnostic ``work`` method remains the explicit FLOP/byte accounting.
        """
        ratio, heads, tp = (int(key[name]) for name in ("compress_ratio", "num_heads", "tp_size"))
        identity = (ratio, heads, tp)
        constants = self._growth_geometry.get(identity)
        if constants is None:
            base = self.work({**key, "phase": "generation", "batch_size": 1,
                              "input_seq_len": 1, "context_len": 0})
            constants = (
                base.math_seconds * 1e6,
                sum(value for name, value in base.bytes.items() if "weights" in name),
                base.bytes["activations"] + base.bytes["rope"],
                4 * heads * self.model.head_dim / self.compute_rates["attention"] * 1e6,
                2 * self.model.deepseek_v4.index_n_heads * self.model.deepseek_v4.index_head_dim
                / self.compute_rates["indexer"] * 1e6 if ratio == 4 else 0,
            )
            self._growth_geometry[identity] = constants
        math_per_token, weights, activation_per_token, attention_per_pair, indexer_per_pair = constants
        batch = max(1, int(key["batch_size"]))
        context = max(0, int(key["context_len"]))
        prefill = key["phase"] == "context"
        queries = max(1, int(key["input_seq_len"])) if prefill else 1
        prefix = context if prefill else max(0, context - 1)
        kv_len = prefix + queries if prefill else prefix
        compressed_len = kv_len // ratio if ratio > 1 else 0
        limit = self.model.deepseek_v4.index_topk if ratio == 4 else compressed_len
        if prefill:
            pairs = causal_limited_pairs(batch, queries, prefix, self.model.sliding_window)
            if ratio > 1:
                pairs += compressed_context_pairs(batch, queries, prefix, ratio, limit)
        else:
            pairs = batch * (min(kv_len, self.model.sliding_window) + min(compressed_len, limit))
        tokens = batch * queries
        indexer_bytes = batch * compressed_len * self.config.indexer_state_bytes if ratio == 4 else 0
        math_us = math_per_token * tokens + attention_per_pair * pairs + indexer_per_pair * tokens * compressed_len
        memory_us = (weights + activation_per_token * tokens + pairs * self.config.kv_state_bytes + indexer_bytes) / self.bandwidth * 1e6
        return math_us, memory_us, indexer_bytes

    def estimate_us(self, key, *, indexer_kv_fraction=1.0):
        """Retain per-query math and sparse KV reads; share only indexer KV.

        ``indexer_kv_fraction`` is Q-block_count/query_rows for a generation
        batch. The source table has one independent request per query row;
        target queries share KV within the kernel's bounded Q-block.
        """
        if not 0 < indexer_kv_fraction <= 1:
            raise ConfigurationError("indexer KV request/query fraction must be in (0, 1]")
        math_us, memory_us, indexer_bytes = self.limits_us(key)
        memory_us -= (1 - indexer_kv_fraction) * indexer_bytes / self.bandwidth * 1e6
        return max(math_us, memory_us)

    def shared_phase_us(self, key, *, indexer_kv_fraction=1.0):
        """Separate full-tensor projections from context-dependent kernel work.

        Input/output and compressor-input GEMMs, index-query/weight projections,
        activation preparation and weight reads are common. Sparse attention,
        index scoring/KV reads and compressor pointwise work remain in the
        phase component. These are separate roofline bounds, not measured
        component latencies; their sum need not equal the module roofline.
        A caller must normalize their allocation to the measured whole module.
        """
        if not 0 < indexer_kv_fraction <= 1:
            raise ConfigurationError("indexer KV request/query fraction must be in (0, 1]")
        ratio, heads, tp = (int(key[name]) for name in ("compress_ratio", "num_heads", "tp_size"))
        identity = (ratio, heads, tp)
        constants = self._shared_geometry.get(identity)
        if constants is None:
            base = self.work({**key, "phase": "generation", "batch_size": 1,
                              "input_seq_len": 1, "context_len": 0})
            pointwise = 0
            if ratio > 1:
                pointwise = 2 * (2 if ratio == 4 else 1) * self.model.head_dim
                if ratio == 4:
                    pointwise += 4 * self.model.deepseek_v4.index_head_dim
            common_math = base.math_seconds * 1e6 - pointwise / self.compute_rates["compressor"] * 1e6
            constants = (common_math,
                         sum(value for name, value in base.bytes.items() if "weights" in name),
                         base.bytes["activations"] + base.bytes["rope"])
            self._shared_geometry[identity] = constants
        math_per_row, weights, activation_per_row = constants
        rows = max(1, int(key["batch_size"])) * (
            max(1, int(key["input_seq_len"])) if key["phase"] == "context" else 1
        )
        common_math = math_per_row * rows
        common_memory = (weights + activation_per_row * rows) / self.bandwidth * 1e6
        math_us, memory_us, indexer_bytes = self.limits_us(key)
        phase_memory = memory_us - common_memory - (1 - indexer_kv_fraction) * indexer_bytes / self.bandwidth * 1e6
        return (max(common_math, common_memory),
                max(0.0, math_us - common_math, phase_memory))

    def work(self, key):
        """Price the source table's uniform q=1 generation or prefill shape.

        Generation context_len includes the new query token, as in the
        collector. Prefill context_len is the already-computed prefix.
        Indexer logits examine all compressed keys; CSA sparse attention is
        capped at top-k, whereas HCA compressed attention is not.
        """
        m, spec, cfg = self.model, self.model.deepseek_v4, self.config
        batch = max(1, int(key["batch_size"]))
        context = max(0, int(key["context_len"]))
        prefill = key["phase"] == "context"
        queries = max(1, int(key["input_seq_len"])) if prefill else 1
        prefix = context if prefill else max(0, context - 1)
        tokens = batch * queries
        kv_len = prefix + queries if prefill else prefix
        ratio = int(key["compress_ratio"])
        heads = int(key["num_heads"])
        groups = spec.o_groups // int(key["tp_size"])
        h, hd, qlr, olr = m.hidden_size, m.head_dim, spec.q_lora_rank, spec.o_lora_rank
        ih, ihd = spec.index_n_heads, spec.index_head_dim

        projection_weights = h * qlr + qlr * heads * hd + h * hd + groups * olr * h
        absorption_weights = heads * hd * olr
        compressor_weights = 0
        compressor_ops = 0
        # ratio=1 is the MTP SWA-only layer; it has no compressor/indexer.
        if ratio > 1:
            mult = 2 if ratio == 4 else 1
            compressor_weights = 2 * h * mult * hd
            compressor_ops = 2 * tokens * compressor_weights + 2 * tokens * mult * hd
            if ratio == 4:
                index_compressor_weights = 2 * h * 2 * ihd
                compressor_weights += index_compressor_weights
                compressor_ops += 2 * tokens * index_compressor_weights + 2 * tokens * 2 * ihd

        compressed_len = kv_len // ratio if ratio > 1 else 0
        compressed_limit = spec.index_topk if ratio == 4 else compressed_len
        if prefill:
            window_pairs = causal_limited_pairs(batch, queries, prefix, m.sliding_window)
            compressed_pairs = compressed_context_pairs(
                batch, queries, prefix, ratio, compressed_limit
            ) if ratio > 1 else 0
        else:
            window_pairs = batch * min(kv_len, m.sliding_window)
            compressed_pairs = batch * min(compressed_len, compressed_limit)
        pairs = window_pairs + compressed_pairs
        indexer_projection_weights = qlr * ih * ihd if ratio == 4 else 0
        indexer_weights = h * ih if ratio == 4 else 0
        indexer_pairs = tokens * compressed_len if ratio == 4 else 0

        flops = {
            "projection": 2 * tokens * projection_weights,
            "output_absorption": 2 * tokens * absorption_weights,
            "compressor": compressor_ops,
            "indexer": 2 * tokens * indexer_projection_weights + 2 * indexer_pairs * ih * ihd,
            "indexer_weights": 2 * tokens * indexer_weights,
            "attention": 4 * heads * hd * pairs,
        }
        memory = {
            "projection_weights": projection_weights * dtype_bytes(cfg.projection_dtype),
            "output_absorption_weights": absorption_weights * dtype_bytes(cfg.output_absorption_dtype),
            "compressor_weights": compressor_weights * dtype_bytes(cfg.compressor_dtype),
            "indexer_projection_weights": indexer_projection_weights * dtype_bytes(cfg.indexer_dtype),
            "indexer_weights": indexer_weights * dtype_bytes("bf16"),
            "activations": tokens * (h + qlr + heads * hd + hd + groups * olr)
            * dtype_bytes(cfg.projection_dtype),
            "rope": tokens * heads * spec.qk_rope_head_dim * dtype_bytes(cfg.attention_dtype),
            "attention_kv": pairs * cfg.kv_state_bytes,
            # SOL assumes reuse across prefill queries in the same request,
            # as AIC does. It is not a measured cache-traffic guarantee.
            "indexer_kv": batch * compressed_len * cfg.indexer_state_bytes if ratio == 4 else 0,
        }
        return AttentionSOLWork(
            flops, memory,
            sum(value / self.compute_rates[name] for name, value in flops.items()),
            sum(memory.values()) / self.bandwidth,
        )

    def describe(self):
        return {
            "model": "DSV4 FLOP/byte roofline anchored to measured boundary efficiency",
            "basis": "AIC dsv4_attention_sol_ms; local grouped FP8 output and BF16 compressors from vLLM",
            "config": asdict(self.config),
            "cache_layout_evidence": "source-table layouts inferred from collector defaults and current vLLM; historical version unverified",
            "source_to_reference_layout_correction": False,
            "compute_flops_per_s": dict(self.compute_rates),
            "memory_bandwidth_bytes_per_s": self.bandwidth,
            "efficiency": "measured(boundary) / SOL(boundary); no fitted throughput multiplier",
        }
