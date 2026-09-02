_GB = 1 << 30

WORKER_ROLES = frozenset({"hybrid", "prefill", "decode"})
WORKER_ROLE_PREFIXES = {"hybrid": "h", "prefill": "p", "decode": "d"}
KV_TRANSFER_ROLES = frozenset({"kv_producer", "kv_consumer", "kv_both"})

AF_PHASES = frozenset({"prefill", "decode"})
AF_PHASE_ORDER = ("prefill", "decode")
WORKER_CAPABILITIES = frozenset({"prefill", "decode_full", "af_attention", "af_ffn"})
WORKER_CAPABILITY_ORDER = (
    "prefill",
    "decode_full",
    "af_attention",
    "af_ffn",
)
LEGACY_ROLE_CAPABILITIES = {
    "prefill": frozenset({"prefill"}),
    "decode": frozenset({"decode_full"}),
    "hybrid": frozenset({"prefill", "decode_full"}),
}
