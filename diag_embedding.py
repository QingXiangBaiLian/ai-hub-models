"""Diagnostic: Compare embedding tables between FP and quantized models.
Also test: what if we use FP embeddings with the quantized ONNX model?
"""
import torch
import math
import numpy as np
from torch.nn import CrossEntropyLoss

# Test tokens from wikitext
WIKITEXT_TOKENS = [271, 283, 8137, 211388, 455, 283, 73714, 8137, 211388, 455, 369, 449, 6163, 4383, 1116, 12069, 321, 32385, 11745, 641]

print("=" * 60)
print("STEP 1: Load FP model and get its embedding")
print("=" * 60)
from qai_hub_models.models.qwen3_5_0_8b.model import (
    Qwen3_5_0_8B,
    Qwen3_5_0_8B_AIMETOnnx,
)

SEQ_LEN = 128
CTX_LEN = 256

fp_model = Qwen3_5_0_8B.from_pretrained(
    sequence_length=SEQ_LEN,
    context_length=CTX_LEN,
)

# Get FP model's embedding weight
fp_embed_weight = fp_model.model.get_input_embeddings().weight.detach()
print(f"FP embed_tokens weight shape: {fp_embed_weight.shape}")
print(f"FP embed_tokens dtype: {fp_embed_weight.dtype}")
print(f"FP embed_tokens stats: min={fp_embed_weight.min():.6f}, max={fp_embed_weight.max():.6f}, std={fp_embed_weight.std():.6f}")

# Get FP model's lm_head weight (to see if tied)
if hasattr(fp_model.model, 'lm_head'):
    fp_lm_head_weight = fp_model.model.lm_head.weight.detach()
    print(f"\nFP lm_head weight shape: {fp_lm_head_weight.shape}")
    print(f"FP lm_head dtype: {fp_lm_head_weight.dtype}")
    tied = torch.equal(fp_embed_weight, fp_lm_head_weight)
    print(f"Tied (embed == lm_head): {tied}")
    if not tied:
        diff = (fp_embed_weight - fp_lm_head_weight).abs()
        print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")

# Get embeddings for wikitext tokens
fp_embeddings = fp_embed_weight[WIKITEXT_TOKENS]
print(f"\nFP embeddings for wikitext tokens shape: {fp_embeddings.shape}")
print(f"FP embeddings stats: min={fp_embeddings.min():.6f}, max={fp_embeddings.max():.6f}")
print(f"FP embedding for token 271: {fp_embeddings[0, :10].tolist()}")
print(f"FP embedding for token 211388: {fp_embeddings[3, :10].tolist()}")

# Save FP model tokenizer before deleting
tokenizer = fp_model.tokenizer
del fp_model
import gc
gc.collect()

print("\n" + "=" * 60)
print("STEP 2: Load quantized model and get its embedding")
print("=" * 60)

model = Qwen3_5_0_8B_AIMETOnnx.from_pretrained(
    host_device=torch.device("cpu"),
    sequence_length=SEQ_LEN,
    context_length=CTX_LEN,
    precision="w4a16",
    checkpoint="/workspace/qwen3_5_0_8b_w4a16_static",
)
model.eval()

# Get quantized model's embedding table
quant_embed = model._get_embedding_table()
quant_embed_weight = quant_embed.weight.detach()
print(f"Quantized embedding table shape: {quant_embed_weight.shape}")
print(f"Quantized embedding dtype: {quant_embed_weight.dtype}")
print(f"Quantized embedding stats: min={quant_embed_weight.min():.6f}, max={quant_embed_weight.max():.6f}, std={quant_embed_weight.std():.6f}")

# Compare with FP
quant_embeddings = quant_embed_weight[WIKITEXT_TOKENS]
print(f"\nQuantized embeddings for wikitext tokens shape: {quant_embeddings.shape}")
print(f"Quantized embedding for token 271: {quant_embeddings[0, :10].tolist()}")
print(f"Quantized embedding for token 211388: {quant_embeddings[3, :10].tolist()}")

# Compare FP vs Quantized embeddings
diff = (fp_embeddings - quant_embeddings).abs()
print(f"\nEmbedding DIFF (FP vs Quantized):")
print(f"  Max diff: {diff.max():.6f}")
print(f"  Mean diff: {diff.mean():.6f}")
print(f"  Token 271 max diff: {diff[0].max():.6f}")
print(f"  Token 211388 max diff: {diff[3].max():.6f}")

# Check cosine similarity
cos_sims = torch.nn.functional.cosine_similarity(fp_embeddings, quant_embeddings, dim=1)
print(f"  Cosine similarities: {cos_sims.tolist()}")

# Also check the ONNX initializer names matching lm_head/embed_tokens
print("\n" + "=" * 60)
print("STEP 3: Check ONNX model initializer names")
print("=" * 60)
import onnx
session = model.quant_sim.session
onnx_model = model.quant_sim.model.model

matching_initializers = []
for weight in onnx_model.graph.initializer:
    if "lm_head" in weight.name or "embed_tokens" in weight.name:
        matching_initializers.append((weight.name, list(weight.dims)))

print(f"Initializers matching 'lm_head' or 'embed_tokens':")
for name, dims in matching_initializers:
    print(f"  {name}: shape={dims}")

# Check which one is actually used
print("\n" + "=" * 60)
print("STEP 4: Run model with FP embeddings vs quantized embeddings")
print("=" * 60)
from qai_hub_models.models._shared.qwen3_5.model import Qwen3_5RopeEmbedding
from qai_hub_models.models._shared.llm.generator import LLM_Generator

embedding_rope = Qwen3_5RopeEmbedding(max_length=CTX_LEN, config=model.llm_config)

# Create input_ids from wikitext (pad to 128)
input_ids = torch.tensor([WIKITEXT_TOKENS + [0] * (SEQ_LEN - len(WIKITEXT_TOKENS))], dtype=torch.long)
attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.int32)
attention_mask[0, len(WIKITEXT_TOKENS):] = 0  # mask padding

print(f"Test input_ids shape: {input_ids.shape}")
print(f"Test attention_mask: {attention_mask[0, :25].tolist()}")

# Test A: Use generator normally (quantized embedding)
generator = LLM_Generator([model], tokenizer, embedding_rope, accumulate_logits_on_cpu=True)
with torch.no_grad():
    outputs_quant = generator(input_ids, attention_mask, None)
logits_quant = outputs_quant.logits
print(f"\nWith QUANTIZED embedding:")
print(f"  Logits shape: {logits_quant.shape}")
print(f"  Logits stats: min={logits_quant.min():.4f}, max={logits_quant.max():.4f}, std={logits_quant.std():.4f}")

# Compute PPL for the valid portion
valid_logits = logits_quant[:, :len(WIKITEXT_TOKENS)-1, :]
valid_labels = input_ids[:, 1:len(WIKITEXT_TOKENS)]
loss_quant = CrossEntropyLoss()(valid_logits.reshape(-1, valid_logits.size(-1)).float(), valid_labels.reshape(-1)).item()
print(f"  Loss: {loss_quant:.4f}, PPL: {math.exp(loss_quant):.2f}")

# Test B: Manually create FP embeddings and pass to model directly
print(f"\nWith FP embeddings (direct model call):")
# Reconstruct fp_embed_weight for the tokens we need
# We already have fp_embeddings from before (saved from FP model)
# Pad to full sequence
fp_input_embeds = torch.zeros(1, SEQ_LEN, 1024, dtype=torch.float32)
fp_input_embeds[0, :len(WIKITEXT_TOKENS), :] = fp_embeddings

# Create the same inputs that the generator would create, but with FP embeddings
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

# Build attention mask the same way as prepare_inputs
padded_attention_mask = torch.cat([
    torch.zeros(1, CTX_LEN - SEQ_LEN),  # KV cache padding
    attention_mask.float(),
], dim=-1)

position_ids = torch.cumsum(padded_attention_mask, dim=1, dtype=torch.int32) - 1
position_ids = position_ids.clip(0, CTX_LEN - 1)
position_ids = position_ids[..., -SEQ_LEN:]

attention_mask_converter = AttentionMaskConverter(True)
cm_attention_mask = attention_mask_converter.to_4d(
    padded_attention_mask,
    query_length=SEQ_LEN,
    key_value_length=CTX_LEN,
    dtype=torch.float32,
)
cm_attention_mask = cm_attention_mask.clip(min=-7100.0)

position_ids_cos, position_ids_sin = embedding_rope.get_embedding(position_ids)

# Get input_spec for dummy past KV
input_specs = model.get_input_spec(
    llm_config=model.llm_config.to_dict(),
    sequence_length=SEQ_LEN,
    context_length=CTX_LEN,
    llm_io_type=model.llm_io_type,
)
padded_past_key_values = [
    torch.zeros(shape)
    for k, (shape, _) in input_specs.items()
    if k.startswith("past_")
]

# Call model with FP embeddings
with torch.no_grad():
    fp_embed_outputs = model(
        fp_input_embeds,
        cm_attention_mask,
        position_ids_cos,
        position_ids_sin,
        *padded_past_key_values,
    )

if isinstance(fp_embed_outputs, (list, tuple)):
    logits_fp_embed = fp_embed_outputs[0]
else:
    logits_fp_embed = fp_embed_outputs

print(f"  Logits shape: {logits_fp_embed.shape}")
print(f"  Logits stats: min={logits_fp_embed.min():.4f}, max={logits_fp_embed.max():.4f}, std={logits_fp_embed.std():.4f}")

valid_logits_fp = logits_fp_embed[:, :len(WIKITEXT_TOKENS)-1, :]
loss_fp = CrossEntropyLoss()(valid_logits_fp.reshape(-1, valid_logits_fp.size(-1)).float(), valid_labels.reshape(-1)).item()
print(f"  Loss: {loss_fp:.4f}, PPL: {math.exp(loss_fp):.2f}")

# Test C: Use quantized embedding but check specific values
print(f"\nWith QUANTIZED embedding (direct model call):")
quant_input_embeds = torch.zeros(1, SEQ_LEN, 1024, dtype=torch.float32)
quant_input_embeds[0, :len(WIKITEXT_TOKENS), :] = quant_embeddings

with torch.no_grad():
    quant_embed_outputs = model(
        quant_input_embeds,
        cm_attention_mask,
        position_ids_cos,
        position_ids_sin,
        *padded_past_key_values,
    )

if isinstance(quant_embed_outputs, (list, tuple)):
    logits_quant_embed = quant_embed_outputs[0]
else:
    logits_quant_embed = quant_embed_outputs

print(f"  Logits shape: {logits_quant_embed.shape}")
print(f"  Logits stats: min={logits_quant_embed.min():.4f}, max={logits_quant_embed.max():.4f}, std={logits_quant_embed.std():.4f}")

valid_logits_q = logits_quant_embed[:, :len(WIKITEXT_TOKENS)-1, :]
loss_q = CrossEntropyLoss()(valid_logits_q.reshape(-1, valid_logits_q.size(-1)).float(), valid_labels.reshape(-1)).item()
print(f"  Loss: {loss_q:.4f}, PPL: {math.exp(loss_q):.2f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Quantized embedding vs FP embedding max diff: {diff.max():.6f}")
print(f"PPL with quantized embedding (via generator): {math.exp(loss_quant):.2f}")
print(f"PPL with FP embedding (direct): {math.exp(loss_fp):.2f}")
print(f"PPL with quantized embedding (direct): {math.exp(loss_q):.2f}")
