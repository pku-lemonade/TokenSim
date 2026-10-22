"""Single-client costs composed from TokenSim rooflines and link models.

No output text, accuracy, host networking, hidden reasoning, or speculative
acceptance is simulated. The scheduled collective option is an uncalibrated
alternative to the current LPX entry point's flat ring, not a runtime change.
"""

from dataclasses import replace
from functools import lru_cache
import math

import numpy as np

from TokenSim.comm.collectives import CollectiveModel, CollectiveQuery
from TokenSim.comm.collective_types import ring_traffic_factor
from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import LinkCatalog
from TokenSim.hardware.device import dtype_bytes
from TokenSim.operator_data.analytical.base import analytical_model_for


class Costs:
    def __init__(
        self,
        hardware,
        model,
        *,
        weight_dtype="bf16",
        kv_dtype="bf16",
        collective="native-ring",
        kernel_us=0.2,
        rho_in=1.0,
        rho_out=1.0,
        chunk=256,
        cross_latency_us=None,
        timing_window="full",
    ):
        if collective not in {"native-ring", "scheduled-direct"}:
            raise ConfigurationError("unknown AA collective hypothesis")
        if weight_dtype not in {"bf16", "fp8"} or kv_dtype not in {"bf16", "fp8"}:
            raise ConfigurationError("AA exploration supports BF16 and FP8 precision")
        if timing_window not in {"full", "tail80"}:
            raise ConfigurationError("unknown output timing window")
        number(kernel_us, "kernel overhead")
        number(rho_in, "input token ratio", positive=True)
        number(rho_out, "output token ratio", positive=True)
        integer(chunk, "prefill chunk", minimum=1)
        if cross_latency_us is not None:
            number(cross_latency_us, "cross latency")
        self.hardware, self.model = hardware, model
        self.weight_dtype, self.kv_dtype = weight_dtype, kv_dtype
        self.collective, self.kernel_us = collective, kernel_us
        self.timing_window = timing_window
        self.rho_in, self.rho_out, self.chunk = rho_in, rho_out, chunk
        self.devices = {
            name: replace(
                hardware.device(name),
                analytical={
                    **hardware.device(name).analytical,
                    "kernel_launch_us": kernel_us,
                },
            )
            for name in ("rubin", "groq_lp30")
        }
        links = LinkCatalog(
            [
                replace(link, half_bandwidth_bytes=0, collective_efficiency=1.0)
                for link in hardware.links
            ]
        )
        self.collectives = {
            device: CollectiveModel(
                hardware.topology(
                    "rubin_nvl72" if device == "rubin" else "groq_lpx256"
                ),
                links,
                launch_us=0,
                fallback="analytical_only",
            )
            for device in self.devices
        }
        self.cross_link = hardware.link("ethernet1600Gb")
        if cross_latency_us is not None:
            self.cross_link = replace(self.cross_link, latency_us=cross_latency_us)

    @lru_cache(maxsize=None)
    def estimate(self, device, table, items):
        spec = self.devices[device]
        return analytical_model_for(spec.family).estimate(table, dict(items), spec)

    def op(self, device, table, **key):
        return self.estimate(device, table, tuple(sorted(key.items())))

    def gemm(self, device, m, n, k):
        return self.op(
            device, "gemm", m=m, n=n, k=k, dtype=self.weight_dtype
        ).latency_us

    def element(self, device, op, tokens, width):
        return self.op(
            device,
            "elementwise",
            op_name=op,
            num_tokens=tokens,
            hidden_size=width,
            dtype="bf16",
        ).latency_us

    @lru_cache(maxsize=None)
    def communication(self, device, tp, tokens=1, operation="all_reduce"):
        if tp == 1:
            return 0.0
        size = tokens * self.model.hidden_size * 2
        if self.collective == "native-ring":
            link = self.hardware.link(self.devices[device].scale_up_link)
            rounds = 2 * (tp - 1) if operation == "all_reduce" else tp - 1
            traffic = ring_traffic_factor(operation, tp) * size
            return (
                rounds * link.latency_us
                + traffic / link.bandwidth_per_direction_bytes_per_s * 1e6
            )
        model = self.collectives[device]
        query = CollectiveQuery(
            operation,
            size,
            model.topology.group_layout(tuple(range(tp))),
            "bf16",
            "direct",
        )
        return model.analytical(query).latency_us

    @lru_cache(maxsize=None)
    def components(self, device, tp, part, full, tokens=1):
        model, h = self.model, self.model.hidden_size
        linear, pointwise, comm = 0.0, 0.0, 0.0
        attention = (
            part in {"full", "attention"}
            or (part == "global_attention" and full)
            or (part == "local_ffn" and not full)
        )
        if attention:
            q, kv, dim = model.dims(full, tp)
            kv_projections = 1 if full and model.global_k_equals_v_projection else 2
            linear += self.gemm(device, tokens, (q + kv_projections * kv) * dim, h)
            linear += self.gemm(device, tokens, h, q * dim)
            # Q/K/V norm, RoPE, two hidden norms and residual are separate
            # kernels. QKV projection is fused. K=V shares the projection only.
            pointwise += self.element(device, "rmsnorm", tokens, h) * 2
            pointwise += self.element(device, "rmsnorm", tokens * q, dim)
            pointwise += self.element(device, "rmsnorm", tokens * kv, dim) * 2
            pointwise += self.element(device, "rope", tokens, (q + kv) * dim)
            pointwise += self.element(device, "residual_add", tokens, h)
            comm += self.communication(device, tp, tokens)
        if part in {"full", "ffn", "local_ffn"}:
            width = model.intermediate_size // tp
            linear += self.gemm(device, tokens, 2 * width, h)
            linear += self.gemm(device, tokens, h, width)
            # GELU plus gate multiplication uses one fused pointwise kernel.
            pointwise += self.element(device, "gelu", tokens, width)
            pointwise += self.element(device, "rmsnorm", tokens, h) * 2
            pointwise += self.element(device, "residual_add", tokens, h)
            comm += self.communication(device, tp, tokens)
        return dict(linear_us=linear, pointwise_us=pointwise, collective_us=comm)

    def attention(self, device, tp, full, *, context, tokens=1):
        q, kv, dim = self.model.dims(full, tp)
        table = "generation_attention" if tokens == 1 else "context_attention"
        key = dict(
            attn_dtype="bf16",
            kv_cache_dtype=self.kv_dtype,
            num_heads=q,
            num_kv_heads=kv,
            head_dim=dim,
            window_size=0,
            batch_size=1,
        )
        if tokens == 1:
            key["context_len"] = context
        else:
            key.update(
                input_seq_len=tokens,
                context_len=context,
                window_size=0 if full else self.model.sliding_window,
            )
        return self.op(device, table, **key)

    def terminal(self, device, tp):
        h, vocab = self.model.hidden_size, self.model.vocab_size
        return (
            self.element(device, "rmsnorm", 1, h)
            + self.gemm(device, 1, math.ceil(vocab / tp), h)
            + self.element(device, "argmax", 1, math.ceil(vocab / tp))
            # Global selected-token reduction, not full logits gathering.
            + (
                self.hardware.link(
                    self.devices[device].scale_up_link
                ).analytical_transfer_us(8)
                if tp > 1
                else 0
            )
        )

    @lru_cache(maxsize=None)
    def decode(self, device, tp, part, inputs, outputs):
        contexts = np.arange(inputs + 1, inputs + outputs, dtype=float)
        result = dict(
            linear_us=0.0,
            pointwise_us=0.0,
            collective_us=0.0,
            attention_us=np.zeros(outputs - 1),
            host_us=0.0,
        )
        for full in (False, True):
            count = sum(
                self.model.full(i) == full for i in range(self.model.num_layers)
            )
            for key, value in self.components(device, tp, part, full).items():
                result[key] += count * value
            attention = (
                part in {"full", "attention"}
                or (part == "global_attention" and full)
                or (part == "local_ffn" and not full)
            )
            if attention:
                keys = (
                    contexts
                    if full
                    else np.minimum(contexts, self.model.sliding_window)
                )
                a = self.attention(device, tp, full, context=1)
                b = self.attention(device, tp, full, context=2)
                compute = a.compute_time_us + (keys - 1) * (
                    b.compute_time_us - a.compute_time_us
                )
                memory = a.memory_time_us + (keys - 1) * (
                    b.memory_time_us - a.memory_time_us
                )
                result["attention_us"] += count * (
                    np.maximum(compute, memory) + a.overhead_us
                )
        if part in {"full", "attention", "global_attention"}:
            result["linear_us"] += self.terminal(device, tp)
        if device == "rubin":
            result["host_us"] = 4.0
        return result

    @lru_cache(maxsize=None)
    def prefill(self, device, tp, inputs):
        total = 0.0
        for start in range(0, inputs, self.chunk):
            tokens = min(self.chunk, inputs - start)
            for full in (False, True):
                count = sum(
                    self.model.full(i) == full for i in range(self.model.num_layers)
                )
                components = self.components(device, tp, "full", full, tokens)
                core = self.attention(
                    device, tp, full, context=start + tokens, tokens=tokens
                )
                total += count * (sum(components.values()) + core.latency_us)
            if device == "rubin":
                total += 4.0
        # Only the final prompt position computes the full output head.
        return total + self.terminal(device, tp)

    def pipeline(self, placement, tokens=1):
        """One-client critical path, including inter-rack PP and token feedback."""
        size = self.model.hidden_size * 2 * tokens
        limit = 72 if placement.device == "rubin" else 256
        local = self.hardware.link(self.devices[placement.device].scale_up_link)
        total = 0.0
        for stage in range(1, len(placement.stages)):
            previous = ((stage - 1) * placement.tp) // limit
            current = (stage * placement.tp) // limit
            link = local if previous == current else self.cross_link
            total += link.analytical_transfer_us(size)
        if len(placement.stages) > 1:
            link = local if placement.cards <= limit else self.cross_link
            total += link.analytical_transfer_us(
                size if placement.head_on_first_stage else 4
            )
        return total

    def af_boundary(self, a_tp, f_tp, layers=None):
        cross = 2 * self.cross_link.analytical_transfer_us(self.model.hidden_size * 2)
        fanout = self.communication("rubin", a_tp, operation="broadcast")
        fanout += self.communication("groq_lp30", f_tp, operation="broadcast")
        return (self.model.num_layers if layers is None else layers) * (cross + fanout)
