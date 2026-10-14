"""Af native fixture."""

from __future__ import annotations

import copy

from TokenSim.config.model_config import ModelSpec
from scripts.deepseek_v3_runtime import configuration


class NativeFixture:
    @classmethod
    def setUpClass(cls):
        cls.base = configuration(a_tp=1, f_tp=1, af_batch_size=8)
        cls.model = ModelSpec.from_mapping(
            dict(
                model_id="qwen3-moe-tiny-random",
                hidden_size=256,
                intermediate_size=512,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                vocab_size=128,
                dtype="bf16",
                kv_cache_dtype="bf16",
                attention_type="gqa",
                moe=dict(
                    num_experts=4,
                    num_experts_per_tok=2,
                    moe_intermediate_size=256,
                    num_shared_experts=0,
                    num_moe_layers=2,
                ),
            )
        )

    def document(self, **overrides):
        doc = copy.deepcopy(self.base.doc)
        doc["af"].update(
            execution_protocol="vllm_afd",
            phases=["decode"],
            communication_timing=dict(enabled=True),
        )
        doc["af"].update(overrides)
        for entry in doc["af"]["links"] + doc["af"]["execution_domains"]:
            entry["phases"] = ["decode"]
        return doc
