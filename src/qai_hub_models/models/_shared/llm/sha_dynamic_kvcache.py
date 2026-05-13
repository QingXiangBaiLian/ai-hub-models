# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import Any

import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:

    class DynamicCache:  # type: ignore[no-redef]
        pass


# This grossly violates underlying type assumptions in DynamicCache, so we
# turn mypy off for this whole file.
class SHADynamicCacheNewValueOnly(DynamicCache):
    """
    Version of DynamicCache that stores the cache as lists for the separate
    heads (so as to avoid concats/splits for SHA) and returning only the
    new values without accumulation.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Ensure key_cache/value_cache always exist, even on transformers
        # versions where DynamicCache uses `layers` instead.
        if not hasattr(self, "key_cache"):
            self.key_cache: list[list[torch.Tensor]] = []  # type: ignore[assignment]
        if not hasattr(self, "value_cache"):
            self.value_cache: list[list[torch.Tensor]] = []  # type: ignore[assignment]

    def update(
        self,
        key_states: list[torch.Tensor],
        value_states: list[torch.Tensor],
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0 and hasattr(self, "_seen_tokens"):
            # self._seen_tokens += key_states.shape[-2]
            # This line is updated
            # Transposed key cache dimensions are [num_heads, batch (always 1), head_dim, sequence_length]
            self._seen_tokens += key_states[0].shape[-1]

        # Update the cache
        if hasattr(self, "key_cache"):
            assert hasattr(self, "key_cache")
            assert hasattr(self, "value_cache")
            if len(self.key_cache) <= layer_idx:
                while len(self.key_cache) < layer_idx:
                    self.key_cache.append(None)  # type: ignore[arg-type]
                    self.value_cache.append(None)  # type: ignore[arg-type]
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                # Do not concatenate the cache, we only need the latest entry
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states

            # Also update self.layers (DynamicLayer objects) if they exist,
            # so that code reading from past_key_values.layers[idx].keys/.values
            # gets properly concatenated tensors.
            if (
                hasattr(self, "layers")
                and layer_idx < len(self.layers)
                and self.layers[layer_idx] is not None
            ):
                if isinstance(key_states, torch.Tensor):
                    k_tensor = key_states
                    v_tensor = value_states
                else:
                    k_tensor = torch.cat(key_states, dim=1)
                    v_tensor = torch.cat(value_states, dim=1)
                self.layers[layer_idx].update(k_tensor, v_tensor, cache_kwargs)

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if len(self.layers) <= layer_idx:
            while len(self.layers) < layer_idx:
                self.layers.append(None)  # type: ignore[arg-type]
            # We are violating the types of the original DynamicCache by using
            # lists
            self.layers.append([key_states, value_states])
        else:
            # Do not concatenate the cache, we only need the latest entry
            self.layers[layer_idx][0] = key_states
            self.layers[layer_idx][1] = value_states

        # return self.key_cache[layer_idx], self.value_cache[layer_idx]
        return self.layers[layer_idx][0], self.layers[layer_idx][1]

    def __len__(self) -> int:
        if hasattr(self, "key_cache"):
            return len(self.key_cache)
        return len(self.layers)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx is None:
            layer_idx = 0
        # Prefer self.layers (DynamicLayer objects) which have accumulated
        # sequence length, over key_cache which only stores the latest entry.
        if hasattr(self, "layers") and len(self.layers) > layer_idx and self.layers[layer_idx] is not None:
            layer = self.layers[layer_idx]
            if hasattr(layer, "get_seq_length"):
                return layer.get_seq_length()
            if hasattr(layer, "keys") and layer.keys is not None:
                return layer.keys.shape[-2]
        if hasattr(self, "key_cache"):
            if len(self.key_cache) <= layer_idx:
                return 0
            if self.key_cache[layer_idx] is None:
                return 0
            # [0] added to get shape since the outermost is list
            # Transposed key cache dimensions are [num_heads, batch (always 1), head_dim, sequence_length]
            return self.key_cache[layer_idx][0].shape[-1]
        if len(self.layers) <= layer_idx:
            return 0
        if self.layers[layer_idx] is None:
            return 0
        # [0] added to get shape since the outermost is list
        # Transposed key cache dimensions are [num_heads, batch (always 1), head_dim, sequence_length]
        return self.layers[layer_idx][0][0].shape[-1]

    def update_conv_state(
        self, conv_state: torch.Tensor, layer_idx: int
    ) -> None:
        """Update conv state for linear attention layers."""
        if not hasattr(self, "conv_states"):
            self.conv_states: list[torch.Tensor] = []  # type: ignore[assignment]
        
        if len(self.conv_states) <= layer_idx:
            while len(self.conv_states) < layer_idx:
                self.conv_states.append(None)  # type: ignore[arg-type]
            self.conv_states.append(conv_state)
        else:
            self.conv_states[layer_idx] = conv_state

    def update_recurrent_state(
        self, recurrent_state: torch.Tensor, layer_idx: int
    ) -> None:
        """Update recurrent state for linear attention layers."""
        if not hasattr(self, "recurrent_states"):
            self.recurrent_states: list[torch.Tensor] = []  # type: ignore[assignment]
        
        if len(self.recurrent_states) <= layer_idx:
            while len(self.recurrent_states) < layer_idx:
                self.recurrent_states.append(None)  # type: ignore[arg-type]
            self.recurrent_states.append(recurrent_state)
        else:
            self.recurrent_states[layer_idx] = recurrent_state
