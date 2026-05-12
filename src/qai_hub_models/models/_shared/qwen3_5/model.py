# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    PositionProcessorBase,
    LLM_AIMETOnnx,
    LLM_QNN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)

# isort: on
import copy
import json
import os
from collections.abc import Collection
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

import qai_hub as hub
from packaging.version import Version
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.qwen3_5 import modeling_qwen3_5

from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.qwen3_5.model_adaptations import (
    QcQwen3_5_apply_rotary_pos_emb,
    QCQwen3_5ForCausalLM,
    QCQwen3_5GatedDeltaNet,
    QCQwen3_5MLP,
    SHAQwen3_5Attention,
    patched_qwen3_5_text_model_forward,
)
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings
from qai_hub_models.utils.base_model import Precision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_qwen"

DATA_DIR = "data"
USE_CACHED_DATA = True

# Qwen3.5 uses the same ChatML format as Qwen3
START_HEADER = "<|im_start|>"
END_HEADER = "<|im_end|>"
SYSTEM_ID = "system"
ASSISTANT_ID = "assistant"
USER_ID = "user"
END_TOKENS = {"<|im_end|>", "<|endoftext|>"}


class Qwen3_5_Optimizations(str, Enum):
    SHA_ATTENTION = "sha_attention"
    RMS_NORM_4_RANK = "rank4_rms_norm"


class Qwen3_5RopeEmbedding:
    """
    Position embedding for Qwen3.5 with partial rotary factor.

    Only partial_rotary_factor of head_dim is used for RoPE,
    so the compact cos/sin have shape (1, 1, seq_len, rotary_dim//2).
    """

    def __init__(
        self,
        max_length: int = 4096,
        config: PretrainedConfig | None = None,
    ) -> None:
        if config is None:
            raise ValueError("config is required for Qwen3_5RopeEmbedding")

        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        rope_params = getattr(config, "rope_parameters", None) or {}
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        rope_theta = rope_params.get("rope_theta", 10000.0)

        rotary_dim = int(head_dim * partial_rotary_factor)

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float)
                / rotary_dim
            )
        )

        # Precompute cos/sin for all positions up to max_length
        positions = torch.arange(max_length, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)  # (max_length, rotary_dim/2)

        # Store in compact format: (1, 1, max_length, rotary_dim//2)
        self.cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        self.sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: (batch_size, sequence_length)
        Returns: (cos, sin) each of shape (batch_size, 1, sequence_length, rotary_dim//2)
        """
        cos = self.cos[0, 0, :, :].to(position_ids.device)
        sin = self.sin[0, 0, :, :].to(position_ids.device)
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


class Qwen3_5Base(LLMBase):
    LMClass = modeling_qwen3_5.Qwen3_5ForCausalLM
    EmbeddingClass = Qwen3_5RopeEmbedding

    # Default prompts for demos
    default_user_prompt = "What is gravity? Keep the answer under ten words."
    default_system_prompt = "You are a helpful AI assistant."

    def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        # Force float32 to avoid dtype mismatch in GatedDeltaNet conv1d.
        # The model config defaults to bfloat16, which causes issues with
        # conv1d ops in linear attention layers during FP evaluation.
        llm_config.torch_dtype = torch.float32
        if hasattr(llm_config, "text_config"):
            llm_config.text_config.torch_dtype = torch.float32
        return llm_config

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        return {
            "global_prefix": "",
            "system_prefix": f"{START_HEADER}{SYSTEM_ID}\n",
            "system_suffix": f"{END_HEADER}\n",
            "user_prefix": f"{START_HEADER}{USER_ID}\n",
            "user_suffix": f"{END_HEADER}\n",
            "assistant_prefix": f"{START_HEADER}{ASSISTANT_ID}\n",
            "assistant_suffix": f"{END_HEADER}\n",
            "default_system_prompt": cls.default_system_prompt,
        }

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        if (
            skip_optimizations
            and Qwen3_5_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        else:
            modeling_qwen3_5.Qwen3_5Attention = SHAQwen3_5Attention  # type: ignore[misc, unused-ignore]

        def bypass_RotaryEmbedding(
            self: modeling_qwen3_5.Qwen3_5TextRotaryEmbedding,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        # Bypass rotary_emb module
        if not hasattr(
            modeling_qwen3_5.Qwen3_5TextRotaryEmbedding, "_original_forward"
        ):
            modeling_qwen3_5.Qwen3_5TextRotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]
                modeling_qwen3_5.Qwen3_5TextRotaryEmbedding.forward
            )
            modeling_qwen3_5.Qwen3_5TextRotaryEmbedding.forward = (
                bypass_RotaryEmbedding
            )
        modeling_qwen3_5.apply_rotary_pos_emb = QcQwen3_5_apply_rotary_pos_emb  # type: ignore[attr-defined, unused-ignore]

        modeling_qwen3_5.Qwen3_5MLP = QCQwen3_5MLP  # type: ignore[misc, unused-ignore]
        modeling_qwen3_5.Qwen3_5ForCausalLM = QCQwen3_5ForCausalLM  # type: ignore[misc, unused-ignore]
        modeling_qwen3_5.Qwen3_5GatedDeltaNet = QCQwen3_5GatedDeltaNet  # type: ignore[misc, unused-ignore]

        # Patch TextModel forward to handle pre-computed (cos, sin) position embeddings
        modeling_qwen3_5.Qwen3_5TextModel.forward = patched_qwen3_5_text_model_forward  # type: ignore[assignment, unused-ignore]

    def _verify_ckpt(self) -> None:
        # llm_config may be the text_config (extracted by get_llm_config),
        # which can have architectures=None and model_type="qwen3_5_text".
        architectures = getattr(self.llm_config, "architectures", None) or []
        arch_ok = len(architectures) == 0 or any(
            arch in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration")
            for arch in architectures
        )
        if not (
            arch_ok
            and self.llm_config.model_type in ("qwen3_5_text", "qwen3_5")
        ):
            raise ValueError(
                "Model config is not compatible with this model implementation."
            )

    def _get_layer_types(self) -> list[str]:
        """Get the layer types from config."""
        if hasattr(self.llm_config, "text_config"):
            config = self.llm_config.text_config
        else:
            config = self.llm_config
        return getattr(config, "layer_types", None) or ["full_attention"] * config.num_hidden_layers

    def _get_text_config(self) -> PretrainedConfig:
        """Get the text sub-config (handles both standalone and VL configs)."""
        if hasattr(self.llm_config, "text_config"):
            return self.llm_config.text_config
        return self.llm_config

    def _get_linear_attn_config(self) -> dict[str, int]:
        """Get linear attention configuration parameters."""
        text_config = self._get_text_config()
        return {
            "linear_conv_kernel_dim": getattr(text_config, "linear_conv_kernel_dim", 4),
            "linear_key_head_dim": getattr(text_config, "linear_key_head_dim", 128),
            "linear_value_head_dim": getattr(text_config, "linear_value_head_dim", 128),
            "linear_num_key_heads": getattr(text_config, "linear_num_key_heads", 16),
            "linear_num_value_heads": getattr(text_config, "linear_num_value_heads", 16),
        }

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *rest: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Forward pass for hybrid Qwen3.5 model with both attention and linear layers.

        Supports two modes based on the number of state tensors provided:

        1. **KV-only mode** (FP eval via generator): state tensors contain only
           KV cache for full_attention layers. Linear attention state is managed
           internally. Outputs only KV cache for full_attention layers.

        2. **Hybrid mode** (ONNX export): state tensors contain entries for ALL
           layers (KV cache for full_attention, conv/recurrent for linear_attention).
           Outputs state for all layers.
        """
        # Unpack position embeddings
        if self.llm_io_type == LLMIOType.huggingface_input_ids:
            position_ids = rest[0]
            state_tensors = rest[1:]
        else:
            position_ids = rest[:2]  # (cos, sin) tuple
            state_tensors = rest[2:]

        layer_types = self._get_layer_types()
        text_config = self._get_text_config()
        linear_attn_config = self._get_linear_attn_config()

        num_full_attention = sum(1 for lt in layer_types if lt == "full_attention")
        num_all_layers = len(layer_types)

        # Detect mode: KV-only (generator) vs hybrid (ONNX export)
        kv_only_mode = len(state_tensors) == num_full_attention * 2
        hybrid_mode = len(state_tensors) == num_all_layers * 2

        if not kv_only_mode and not hybrid_mode:
            raise ValueError(
                f"Expected {num_full_attention * 2} (KV-only) or "
                f"{num_all_layers * 2} (hybrid) state tensors, "
                f"got {len(state_tensors)}."
            )

        # Build DynamicCache with proper layer structure
        cache = DynamicCache(config=text_config)

        # Pre-populate cache with input state
        tensor_idx = 0
        kv_tensor_idx = 0
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                # KV cache: past_key shape (num_kv_heads, 1, head_dim, cache_len)
                #           past_value shape (num_kv_heads, 1, cache_len, head_dim)
                if kv_only_mode:
                    past_key = state_tensors[kv_tensor_idx]
                    past_value = state_tensors[kv_tensor_idx + 1]
                    kv_tensor_idx += 2
                else:
                    past_key = state_tensors[tensor_idx]
                    past_value = state_tensors[tensor_idx + 1]
                    tensor_idx += 2

                # Convert from SHA format to standard HF format:
                # (num_kv_heads, 1, head_dim, cache_len) -> (1, num_kv_heads, cache_len, head_dim)
                k = past_key.permute(1, 0, 3, 2)
                v = past_value.permute(1, 0, 2, 3)
                cache.update(k, v, layer_idx)
            else:
                if hybrid_mode:
                    # Linear attention: conv_state and recurrent_state
                    conv_state = state_tensors[tensor_idx]
                    recurrent_state = state_tensors[tensor_idx + 1]
                    tensor_idx += 2
                else:
                    # KV-only mode: use internally saved state or zeros
                    if hasattr(self, "_linear_attn_cache") and layer_idx in self._linear_attn_cache:
                        conv_state, recurrent_state = self._linear_attn_cache[layer_idx]
                    else:
                        conv_kernel_dim = linear_attn_config["linear_conv_kernel_dim"]
                        key_dim = (
                            linear_attn_config["linear_num_key_heads"]
                            * linear_attn_config["linear_key_head_dim"]
                        )
                        value_dim = (
                            linear_attn_config["linear_num_value_heads"]
                            * linear_attn_config["linear_value_head_dim"]
                        )
                        conv_dim = key_dim * 2 + value_dim
                        num_v_heads = linear_attn_config["linear_num_value_heads"]
                        k_head_dim = linear_attn_config["linear_key_head_dim"]
                        v_head_dim = linear_attn_config["linear_value_head_dim"]

                        conv_state = torch.zeros(
                            1, conv_dim, conv_kernel_dim - 1,
                            device=input_tokens.device,
                            dtype=torch.float32,
                        )
                        recurrent_state = torch.zeros(
                            1, num_v_heads, k_head_dim, v_head_dim,
                            device=input_tokens.device,
                            dtype=torch.float32,
                        )
                cache.update_conv_state(conv_state, layer_idx)
                cache.update_recurrent_state(recurrent_state, layer_idx)

        # Run model
        model_kwargs: dict[str, Any] = {
            self.main_input_name: input_tokens,
            "attention_mask": self.attention_mask_multiplier * attention_mask,
            "position_ids": position_ids,
            "past_key_values": cache,
        }
        out = self.model(**model_kwargs)

        # Extract output states
        out_cache = out["past_key_values"]
        flat_output_states: list[torch.Tensor] = []

        # Save linear attention state internally for KV-only mode
        if kv_only_mode:
            if not hasattr(self, "_linear_attn_cache"):
                self._linear_attn_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                # Extract KV cache output (only new tokens)
                if hasattr(out_cache, "key_cache"):
                    keys = out_cache.key_cache[layer_idx]
                    values = out_cache.value_cache[layer_idx]
                else:
                    keys = out_cache.layers[layer_idx].keys
                    values = out_cache.layers[layer_idx].values

                # Convert to SHA output format:
                # (1, num_kv_heads, seq_len, head_dim) -> (num_kv_heads, 1, head_dim, seq_len)
                k_out = keys[:, :, -self.sequence_length:, :].permute(1, 0, 3, 2)
                v_out = values[:, :, -self.sequence_length:, :].permute(1, 0, 2, 3)
                flat_output_states.append(k_out)
                flat_output_states.append(v_out)
            else:
                # Extract linear attention state
                layer_cache = out_cache.layers[layer_idx]
                conv_state_out = layer_cache.conv_states
                recurrent_state_out = layer_cache.recurrent_states

                if kv_only_mode:
                    # Save internally, don't include in output
                    self._linear_attn_cache[layer_idx] = (
                        conv_state_out.detach(),
                        recurrent_state_out.detach(),
                    )
                else:
                    flat_output_states.append(conv_state_out)
                    flat_output_states.append(recurrent_state_out)

        return [out["logits"], *flat_output_states]

    @staticmethod
    def _get_output_names(
        num_hidden_layers: int,
        layer_types: list[str] | None = None,
        kv_only: bool = False,
    ) -> list[str]:
        """
        Generate output names for the hybrid model.

        For full_attention layers: past_key_{i}_out, past_value_{i}_out
        For linear_attention layers: conv_state_{i}_out, recurrent_state_{i}_out

        If kv_only=True, only include outputs for full_attention layers (for FP eval).
        """
        output_names = ["logits"]
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                output_names.append(f"past_key_{i}_out")
                output_names.append(f"past_value_{i}_out")
            elif not kv_only:
                output_names.append(f"conv_state_{i}_out")
                output_names.append(f"recurrent_state_{i}_out")
        return output_names

    @staticmethod
    def _get_input_spec_hybrid(
        num_hidden_layers: int,
        sequence_length: int,
        context_length: int,
        hidden_size: int,
        num_key_value_heads: int,
        num_attention_heads: int,
        head_dim: int,
        layer_types: list[str],
        linear_attn_config: dict[str, int],
        partial_rotary_factor: float = 0.25,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
        kv_only: bool = False,
    ) -> InputSpec:
        """
        Build input spec for hybrid model with both full_attention and linear_attention layers.

        If kv_only=True, only include KV cache entries for full_attention layers (for FP eval).
        Linear attention state is managed internally by the model in that case.
        """
        rotary_dim = int(head_dim * partial_rotary_factor)
        embed_dim = rotary_dim // 2
        input_spec: InputSpec = {}

        # Primary input
        if llm_io_type == LLMIOType.genie_input_embeds:
            input_spec["input_embeds"] = ((1, sequence_length, hidden_size), "float32")
        else:
            input_spec["input_ids"] = ((1, sequence_length), "int32")

        # Attention mask
        input_spec["attention_mask"] = (
            (1, 1, sequence_length, context_length),
            "float32",
        )

        # Position IDs
        if llm_io_type == LLMIOType.huggingface_input_ids:
            input_spec["position_ids"] = ((1, sequence_length), "int32")
        else:
            input_spec["position_ids_cos"] = (
                (1, 1, sequence_length, embed_dim),
                "float32",
            )
            input_spec["position_ids_sin"] = (
                (1, 1, sequence_length, embed_dim),
                "float32",
            )

        # Per-layer state inputs
        conv_kernel_dim = linear_attn_config["linear_conv_kernel_dim"]
        key_dim = (
            linear_attn_config["linear_num_key_heads"]
            * linear_attn_config["linear_key_head_dim"]
        )
        value_dim = (
            linear_attn_config["linear_num_value_heads"]
            * linear_attn_config["linear_value_head_dim"]
        )
        conv_dim = key_dim * 2 + value_dim
        num_v_heads = linear_attn_config["linear_num_value_heads"]
        k_head_dim = linear_attn_config["linear_key_head_dim"]
        v_head_dim = linear_attn_config["linear_value_head_dim"]

        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                # Standard KV cache
                input_spec[f"past_key_{i}_in"] = (
                    (num_key_value_heads, 1, head_dim, context_length - sequence_length),
                    "float32",
                )
                input_spec[f"past_value_{i}_in"] = (
                    (num_key_value_heads, 1, context_length - sequence_length, head_dim),
                    "float32",
                )
            elif not kv_only:
                # GatedDeltaNet state (only for ONNX export, not FP eval)
                input_spec[f"conv_state_{i}_in"] = (
                    (1, conv_dim, conv_kernel_dim - 1),
                    "float32",
                )
                input_spec[f"recurrent_state_{i}_in"] = (
                    (1, num_v_heads, k_head_dim, v_head_dim),
                    "float32",
                )

        return input_spec


class Qwen3_5PositionProcessor(PositionProcessorBase):
    """Prepares positions (RopeEmbedding and attention mask); used by ORT GenAI."""

    def __init__(
        self,
        context_length: int,
        config: PretrainedConfig,
    ) -> None:
        super().__init__(context_length, config=config)
        self.context_len = context_length
        self.rope_embedding = Qwen3_5RopeEmbedding(max_length=self.context_len, config=config)

    def forward(
        self, attention_mask_before_processor: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position_ids_cos, position_ids_sin = self.rope_embedding.get_embedding(
            position_ids
        )
        attention_mask_converter = AttentionMaskConverter(True)
        attention_mask = attention_mask_converter.to_4d(
            attention_mask_before_processor,
            query_length=position_ids.shape[1],
            key_value_length=attention_mask_before_processor.shape[1],
            dtype=torch.float32,
        )
        attention_mask = attention_mask.clip(-50, 0)
        return attention_mask, position_ids_cos, position_ids_sin


class Qwen3_5Base_AIMETOnnx(LLM_AIMETOnnx):
    EmbeddingClass = Qwen3_5RopeEmbedding
    FPModel = Qwen3_5Base

    ada_scale_model_type: str | None = "qwen3_5"

    def __init__(
        self,
        quant_sim: QuantizationSimModel,
        host_device: torch.device,
        checkpoint: str | os.PathLike | Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        llm_config: PretrainedConfig | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
    ) -> None:
        super().__init__(
            quant_sim=quant_sim,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            attention_mask_min_clip=attention_mask_min_clip,
            attention_mask_multiplier=attention_mask_multiplier,
        )

    @staticmethod
    def _get_output_names(
        num_hidden_layers: int,
        layer_types: list[str] | None = None,
    ) -> list[str]:
        output_names = ["logits"]
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                output_names.append(f"past_key_{i}_out")
                output_names.append(f"past_value_{i}_out")
            else:
                output_names.append(f"conv_state_{i}_out")
                output_names.append(f"recurrent_state_{i}_out")
        return output_names

    @classmethod
    def prepare_genie_assets(
        cls,
        hub_device: hub.Device,
        checkpoint: str | os.PathLike | Path,
        llm_config: PretrainedConfig,
        context_lengths: list[int],
        model_list: list[str],
        output_path: Path,
        precision: Precision,
        encodings_path: str | os.PathLike | Path,
        input_specs: dict[str, Any],
        output_specs: dict[str, Any],
        model_id: str,
        model_name: str,
    ) -> None:
        super().prepare_genie_assets(
            hub_device,
            checkpoint,
            llm_config,
            context_lengths,
            model_list,
            output_path,
            precision,
            encodings_path,
            input_specs,
            output_specs,
            model_id=model_id,
            model_name=model_name,
        )

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *rest: torch.Tensor,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        return super().forward(
            input_tokens,
            self.attention_mask_multiplier * attention_mask,
            *rest,
        )

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """Make sure AIMET encodings are ready for ONNX split."""
        with open(src_encodings_path) as f:
            encodings = json.load(f)

        model = onnx.load(onnx_model_path)

        model_input_names = {}
        for node in model.graph.node:
            model_input_names[node.name] = node.input

        uses_lists = Version(encodings["version"]) >= Version("1.0.0")
        assert uses_lists

        # Convert encodings to dictionaries for faster look-ups
        encodings["activation_encodings"] = {
            v["name"]: v for v in encodings["activation_encodings"]
        }
        encodings["param_encodings"] = {
            v["name"]: v for v in encodings["param_encodings"]
        }

        # Propagate embedding encodings
        embed_a_name = "/model/model/embed_tokens/Gather_output_0"
        embed_w_name = "model.model.embed_tokens.weight"
        encodings["activation_encodings"][embed_a_name] = copy.deepcopy(
            encodings["activation_encodings"][embed_w_name]
        )
        for key in encodings["activation_encodings"]:
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(
                    encodings["activation_encodings"][key]
                )

        encodings["activation_encodings"][embed_a_name]["name"] = embed_a_name

        propagate_memory_encodings(encodings, model)

        # Convert back
        encodings["activation_encodings"] = list(
            encodings["activation_encodings"].values()
        )
        encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)


class Qwen3_5Base_QNN(LLM_QNN):
    FPModel = Qwen3_5Base
    EmbeddingClass = Qwen3_5RopeEmbedding
    num_layers_per_split: int

    @staticmethod
    def _get_output_names(
        num_hidden_layers: int,
        layer_types: list[str] | None = None,
    ) -> list[str]:
        output_names = ["logits"]
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        for i, layer_type in enumerate(layer_types):
            if layer_type == "full_attention":
                output_names.append(f"past_key_{i}_out")
                output_names.append(f"past_value_{i}_out")
            else:
                output_names.append(f"conv_state_{i}_out")
                output_names.append(f"recurrent_state_{i}_out")
        return output_names
