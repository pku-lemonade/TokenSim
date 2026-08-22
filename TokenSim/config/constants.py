_GB = 1 << 30

WORKER_ROLES = frozenset({"hybrid", "prefill", "decode"})
WORKER_ROLE_PREFIXES = {"hybrid": "h", "prefill": "p", "decode": "d"}
KV_TRANSFER_ROLES = frozenset({"kv_producer", "kv_consumer", "kv_both"})
