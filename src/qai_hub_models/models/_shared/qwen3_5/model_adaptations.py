# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F
import transformers
from packaging.version import Version
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
)

from qai_hub_models.models._shared.llm.common import TORCH_SUPPORTS_DYNAMIC_SHAPE
from qai_hub_models.models._shared.llm.model_adaptations import (
    ConvInplaceLinear,
    repeat_kv,
)


def _apply_rope_single_partial(
    x: torch.Tensor, rope_vals: tuple[torch.Tensor, torch.Tensor], rotary_dim: int
) -> torch.Tensor:
    """
    Apply rotary position embeddings to partial dimensions of the input.

    For Qwen3.5 with partial_rotary_factor=0.25, only 64 out of 256 dims use RoPE.
    The compact rope_vals have shape (1, 1, seqlen, rotary_dim//2).
    """
    rope_real = rope_vals[0]  # (1, 1, seqlen, rotary_dim//2)
    rope_im = rope_vals[1]  # (1, 1, seqlen, rotary_dim//2)

    # Split into rotary and passthrough parts
    x_rot = x[:, :, :, :rotary_dim]
    x_pass = x[:, :, :, rotary_dim:]

    half_rot = rotary_dim // 2
    x_real = x_rot[:, :, :, :half_rot]
    x_im = x_rot[:, :, :, half_rot:]

    x_prod_real = x_real * rope_real - x_im * rope_im
    x_prod_im = x_real * rope_im + x_im * rope_real

    x_rot_out = torch.cat((x_prod_real, x_prod_im), dim=3)
    return torch.cat((x_rot_out, x_pass), dim=3)


def QcQwen3_5_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: list[int] | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply partial rotary position embedding for Qwen3.5.

    cos/sin are in compact format: (1, 1, seq_len, rotary_dim//2).
    rotary_dim = head_dim * partial_rotary_factor.
    """
    rotary_dim = cos.shape[-1] * 2
    query_states = _apply_rope_single_partial(q, (cos, sin), rotary_dim)
    key_states = _apply_rope_single_partial(k, (cos, sin), rotary_dim)
    return query_states, key_states


class SHAQwen3_5Attention(Qwen3_5Attention):
    """Split-Head Attention version of Qwen3.5 Attention (with Convs and gating).

    Key differences from Qwen3:
    - q_proj outputs 2x (query + gate), gate is applied via sigmoid after attention
    - Partial rotary embeddings (only partial_rotary_factor of head_dim)
    - q_norm and k_norm applied per-head
    - Uses (1+weight) style RMSNorm (Qwen3NextRMSNorm)
    """

    def prepare_conv(self) -> None:
        if not hasattr(self, "forward_no_conv"):
            # q_proj outputs num_heads * head_dim * 2 (query + gate)
            self.q_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.config.num_attention_heads * self.head_dim * 2,
                1,
                bias=self.q_proj.bias is not None,
            )
            self.k_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                1,
                bias=self.k_proj.bias is not None,
            )
            self.v_proj_conv = nn.Conv2d(
                self.config.hidden_size,
                self.config.num_key_value_heads * self.head_dim,
                1,
                bias=self.v_proj.bias is not None,
            )
            self.o_proj_conv = nn.Conv2d(
                self.config.num_attention_heads * self.head_dim,
                self.config.hidden_size,
                1,
                bias=self.o_proj.bias is not None,
            )

            self.q_proj_conv.weight.data.copy_(self.q_proj.weight[:, :, None, None])
            self.k_proj_conv.weight.data.copy_(self.k_proj.weight[:, :, None, None])
            self.v_proj_conv.weight.data.copy_(self.v_proj.weight[:, :, None, None])
            self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])

            if self.q_proj.bias is not None:
                assert self.q_proj_conv.bias is not None
                self.q_proj_conv.bias.data.copy_(self.q_proj.bias)
            if self.k_proj.bias is not None:
                assert self.k_proj_conv.bias is not None
                self.k_proj_conv.bias.data.copy_(self.k_proj.bias)
            if self.v_proj.bias is not None:
                assert self.v_proj_conv.bias is not None
                self.v_proj_conv.bias.data.copy_(self.v_proj.bias)
            if self.o_proj.bias is not None:
                assert self.o_proj_conv.bias is not None
                self.o_proj_conv.bias.data.copy_(self.o_proj.bias)

            del self.q_proj
            del self.k_proj
            del self.v_proj
            del self.o_proj

    def prepare_sha(self) -> None:
        if not (
            hasattr(self, "q_proj_conv")
            and hasattr(self, "k_proj_conv")
            and hasattr(self, "o_proj_conv")
            and hasattr(self, "v_proj_conv")
        ):
            raise RuntimeError(
                "The method 'prepare_sha' cannot be run on model without running 'prepare_conv' first."
            )

        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads

        if not hasattr(self, "forward_mha"):
            # Each q head outputs head_dim * 2 (query + gate)
            self.q_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.config.hidden_size,
                        self.head_dim * 2,
                        1,
                        bias=self.q_proj_conv.bias is not None,
                    )
                    for _ in range(num_heads)
                ]
            )
            self.k_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.config.hidden_size,
                        self.head_dim,
                        1,
                        bias=self.k_proj_conv.bias is not None,
                    )
                    for _ in range(num_kv_heads)
                ]
            )
            self.v_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.config.hidden_size,
                        self.head_dim,
                        1,
                        bias=self.v_proj_conv.bias is not None,
                    )
                    for _ in range(num_kv_heads)
                ]
            )

            # Per-head q_norm and k_norm (Qwen3NextRMSNorm style: (1 + weight) * norm)
            self.q_norm_sha = nn.ModuleList(
                [
                    Qwen3_5RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
                    for _ in range(num_heads)
                ]
            )
            self.k_norm_sha = nn.ModuleList(
                [
                    Qwen3_5RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
                    for _ in range(num_kv_heads)
                ]
            )

            # Copy weights from original q_norm/k_norm (shared weights)
            for i in range(num_heads):
                q_norm = self.q_norm_sha[i]
                assert isinstance(q_norm, Qwen3_5RMSNorm)
                q_norm.weight.data.copy_(self.q_norm.weight.data)
            for i in range(num_kv_heads):
                k_norm = self.k_norm_sha[i]
                assert isinstance(k_norm, Qwen3_5RMSNorm)
                k_norm.weight.data.copy_(self.k_norm.weight.data)

            self.forward_mha = cast(
                Callable[
                    [
                        torch.Tensor,
                        torch.Tensor | None,
                        torch.LongTensor | None,
                        Cache | None,
                        bool,
                        bool,
                        torch.LongTensor | None,
                        Any,
                    ],
                    tuple[
                        torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None
                    ],
                ],
                self.forward,
            )
            self.forward = self.forward_sha  # type: ignore[assignment, unused-ignore]

        # Copy q_proj weights (head_dim * 2 per head)
        for i in range(num_heads):
            start_idx = i * self.head_dim * 2
            end_idx = (i + 1) * self.head_dim * 2
            q_proj = self.q_proj_sha[i]
            assert isinstance(q_proj, (nn.Linear, nn.Conv2d))
            q_proj.weight.data.copy_(self.q_proj_conv.weight[start_idx:end_idx, :])
            if self.q_proj_conv.bias is not None and q_proj.bias is not None:
                q_proj.bias.data.copy_(self.q_proj_conv.bias[start_idx:end_idx])

        # Copy k_proj and v_proj weights
        for i in range(num_kv_heads):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            k_proj = self.k_proj_sha[i]
            v_proj = self.v_proj_sha[i]
            assert isinstance(k_proj, (nn.Linear, nn.Conv2d))
            assert isinstance(v_proj, (nn.Linear, nn.Conv2d))
            k_proj.weight.data.copy_(self.k_proj_conv.weight[start_idx:end_idx, :])
            v_proj.weight.data.copy_(self.v_proj_conv.weight[start_idx:end_idx, :])
            if self.k_proj_conv.bias is not None and k_proj.bias is not None:
                k_proj.bias.data.copy_(self.k_proj_conv.bias[start_idx:end_idx])
            if self.v_proj_conv.bias is not None and v_proj.bias is not None:
                v_proj.bias.data.copy_(self.v_proj_conv.bias[start_idx:end_idx])

        del self.q_proj_conv
        del self.k_proj_conv
        del self.v_proj_conv
        del self.q_norm
        del self.k_norm

    def forward_sha(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()
        hidden_size = self.config.hidden_size
        num_kv_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )

        partial_rotary_factor = self.config.rope_parameters.get(
            "partial_rotary_factor", 1.0
        )
        rotary_dim = int(self.head_dim * partial_rotary_factor)

        # Use past_key_values as the cache variable
        past_key_value = past_key_values

        if TORCH_SUPPORTS_DYNAMIC_SHAPE:
            hidden_states = hidden_states.unsqueeze(2)
        else:
            hidden_states = torch.reshape(hidden_states, (bsz, -1, 1, hidden_size))
        hidden_states = hidden_states.transpose(1, 3)

        # Project Q (with gate), K, V and apply norms
        # q_proj_sha outputs head_dim*2, split into query and gate
        query_states = []
        gate_states = []
        for q_proj, q_norm in zip(self.q_proj_sha, self.q_norm_sha, strict=False):
            qg = q_proj(hidden_states).permute(0, 2, 3, 1)  # (B, 1, seq, head_dim*2)
            q, g = qg.chunk(2, dim=-1)  # Each (B, 1, seq, head_dim)
            q = q_norm(q)
            query_states.append(q)
            gate_states.append(g)

        key_states = [
            k_norm(k_proj(hidden_states).permute(0, 2, 3, 1))
            for k_proj, k_norm in zip(self.k_proj_sha, self.k_norm_sha, strict=False)
        ]
        value_states = [
            v_proj(hidden_states).permute(0, 2, 3, 1) for v_proj in self.v_proj_sha
        ]

        kv_seq_len = value_states[0].shape[-2]

        assert position_embeddings is not None
        # Apply partial rotary embeddings
        query_states = [
            _apply_rope_single_partial(q, position_embeddings, rotary_dim)
            for q in query_states
        ]
        key_states = [
            _apply_rope_single_partial(k, position_embeddings, rotary_dim)
            for k in key_states
        ]

        if past_key_value is not None:
            transposed_key_states = [
                key_state.transpose(2, 3) for key_state in key_states
            ]

            # Stack per-head lists into single tensors for cache storage.
            # keys: list of (B, 1, head_dim, seq) -> (B, KV_heads, seq, head_dim)
            stacked_keys = torch.cat(transposed_key_states, dim=1).transpose(-1, -2)
            # values: list of (B, 1, seq, head_dim) -> (B, KV_heads, seq, head_dim)
            stacked_values = torch.cat(value_states, dim=1)

            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # update() stores new KV, concatenates with any existing cache,
            # and returns the full accumulated tensors.
            full_keys, full_values = past_key_value.update(
                stacked_keys,
                stacked_values,
                self.layer_idx,
                cache_kwargs,
            )

            # Unpack accumulated cache back to per-head lists.
            # full_keys: (B, KV_heads, total_seq, head_dim) -> transpose + split
            key_states = list(full_keys.transpose(-1, -2).split(1, dim=1))
            # full_values: (B, KV_heads, total_seq, head_dim) -> split
            value_states = list(full_values.split(1, dim=1))
            kv_seq_len = full_values.shape[-2]
        else:
            key_states = [
                key_state.transpose(2, 3) for key_state in key_states
            ]

        key_states = list(repeat_kv(key_states, num_kv_groups))
        value_states = list(repeat_kv(value_states, num_kv_groups))

        attn_weights = [
            torch.matmul(q, k / math.sqrt(self.head_dim))
            for q, k in zip(query_states, key_states, strict=False)
        ]
        if attn_weights[0].size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, 1, q_len, kv_seq_len)}, but is"
                f" {attn_weights[0].size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = [aw + attention_mask for aw in attn_weights]

        # Upcast attention to fp32
        attn_weights = [
            nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(
                query_states[0].dtype
            )
            for aw in attn_weights
        ]
        attn_weights = [
            nn.functional.dropout(aw, p=self.attention_dropout, training=self.training)
            for aw in attn_weights
        ]
        attn_output = [
            torch.matmul(aw, v)
            for aw, v in zip(attn_weights, value_states, strict=False)
        ]

        if attn_output[0].size() != (bsz, 1, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, 1, q_len, self.head_dim)}, but is"
                f" {attn_output[0].size()}"
            )

        # Apply gate: output = attn * sigmoid(gate)
        attn_output = [
            ao * torch.sigmoid(g)
            for ao, g in zip(attn_output, gate_states, strict=False)
        ]

        attn_output_return: torch.Tensor = torch.cat(attn_output, dim=3)
        attn_output_return = attn_output_return.permute(0, 3, 1, 2)
        attn_output_return = self.o_proj_conv(attn_output_return)
        attn_output_return = attn_output_return.transpose(1, 3)
        if TORCH_SUPPORTS_DYNAMIC_SHAPE:
            attn_output_return = attn_output_return.squeeze(2)
        else:
            attn_output_return = attn_output_return.reshape(bsz, q_len, hidden_size)

        attn_weights_return = attn_weights if output_attentions else None

        assert Version(transformers.__version__) >= Version("4.48.0")
        return attn_output_return, attn_weights_return


class QCQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """
    Adapted GatedDeltaNet for explicit state I/O (needed for ONNX export).

    Instead of relying on DynamicCache for in-place state updates,
    this module accepts conv_state and recurrent_state as explicit inputs
    and returns updated states as outputs.
    """

    def forward_explicit_state(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with explicit state tensors.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            conv_state: (batch, conv_dim, kernel_size - 1)
            recurrent_state: (batch, num_v_heads, k_head_dim, v_head_dim)
            attention_mask: optional (batch, seq_len) bool/float mask

        Returns:
            output: (batch, seq_len, hidden_size)
            new_conv_state: (batch, conv_dim, kernel_size - 1)
            new_recurrent_state: (batch, num_v_heads, k_head_dim, v_head_dim)
        """
        hidden_states = _apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        # Project QKV
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, proj_dim, seq_len)

        # Project z, b, a
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Apply causal conv1d with explicit state
        # Concatenate conv_state with current input
        conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)
        # Update conv state: keep last (kernel_size - 1) elements
        new_conv_state = conv_input[:, :, -(self.conv_kernel_size - 1):]

        # Apply conv1d
        mixed_qkv = F.silu(
            F.conv1d(
                conv_input,
                self.conv1d.weight,
                self.conv1d.bias,
                padding=0,
                groups=mixed_qkv.shape[1],
            )[:, :, -seq_len:]
        )

        mixed_qkv = mixed_qkv.transpose(1, 2)  # (B, seq_len, proj_dim)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # Compute decay
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # Use torch fallback for delta rule (ONNX-compatible)
        core_attn_out, new_recurrent_state = _torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        # Apply output norm with gating
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output, new_conv_state, new_recurrent_state


class QCQwen3_5MLP(Qwen3_5MLP):
    def prepare_conv(self) -> None:
        self.down_proj = ConvInplaceLinear(self.down_proj)  # type: ignore[has-type, arg-type, unused-ignore]


class QCQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def prepare_conv(self) -> None:
        self.lm_head = ConvInplaceLinear(self.lm_head)  # type: ignore[has-type, arg-type, unused-ignore]


def patched_qwen3_5_text_model_forward(
    self: Qwen3_5TextModel,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: Any = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs: Any,
) -> Any:
    """
    Patched forward for Qwen3_5TextModel that handles pre-computed (cos, sin) position embeddings.

    When position_ids is a tuple of (cos, sin), skip M-RoPE processing and use directly.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ModelOutputWithPast,
        create_causal_mask,
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    # Detect if position_ids is already pre-computed (cos, sin) tuple
    if isinstance(position_ids, (tuple, list)) and len(position_ids) == 2:
        # Pre-computed position embeddings from the framework
        position_embeddings = tuple(position_ids)
        text_position_ids = None

        # Create causal mask
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        # For linear attention, use 2D attention mask
        linear_attn_mask = self._update_linear_attn_mask(
            attention_mask, past_key_values
        )
    else:
        # Standard M-RoPE path
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            position_ids = (
                torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                )
                + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(
                4, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                4, position_ids.shape[0], -1
            )

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(
            attention_mask, past_key_values
        )

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

    hidden_states = inputs_embeds

    for i, decoder_layer in enumerate(
        self.layers[: self.config.num_hidden_layers]
    ):
        layer_mask = (
            linear_attn_mask
            if self.config.layer_types[i] == "linear_attention"
            else causal_mask
        )

        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def _apply_mask_to_padding_states(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    """Tunes out the hidden states for padding tokens."""
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization aligned with FLA library implementation."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pure PyTorch implementation of the chunked gated delta rule.
    ONNX-exportable fallback for the FLA kernel.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # Reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # Chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = (
        (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()
    ).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            dtype=value.dtype, device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # Process each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
