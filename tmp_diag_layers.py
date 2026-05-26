"""Identify which layer groups cause the most INT4 degradation."""
import sys, os
sys.path.insert(0, "/workspace/ai-hub-models/src")
import torch
import numpy as np

from qai_hub_models.models.qwen3_5_0_8b import Model
from qai_hub_models.models._shared.llm.generator import LLM_Generator
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B as FP_Model

print("Loading quantized model...")
quant_model = Model.from_pretrained(
    checkpoint="/workspace/qwen3_5_0_8b_w4a16_static",
    sequence_length=128,
    context_length=256,
    host_device=torch.device("cpu"),
).to("cpu")

tokenizer = quant_model.tokenizer
text = "The capital of France is"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

EmbeddingClass = FP_Model.EmbeddingClass
embedding = EmbeddingClass(max_length=256, config=quant_model.llm_config)
generator = LLM_Generator([quant_model], tokenizer, embedding, accumulate_logits_on_cpu=True)

quant_sim = quant_model.quant_sim

# First get FP baseline (all quantizers disabled)
print("Getting FP baseline...")
all_enabled = []
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if qc_op.enabled:
        all_enabled.append(name)
        qc_op.enabled = False

quant_model._linear_attn_cache.clear()
out_fp = generator(input_ids, torch.ones_like(input_ids))
logits_fp = out_fp.logits.clone()
print(f"FP baseline top: {tokenizer.decode([logits_fp[0,-1].argmax()])!r}")

# Helper to compute cosine sim
def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

# Categorize quantizers by layer/component
layer_types = ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]

# Map quantizer names to categories
param_quantizers = [n for n in all_enabled if n not in quant_sim.activation_names]
act_quantizers = [n for n in all_enabled if n in quant_sim.activation_names]

print(f"\nTotal enabled quantizers: {len(all_enabled)}")
print(f"  Param quantizers: {len(param_quantizers)}")
print(f"  Activation quantizers: {len(act_quantizers)}")

# Group param quantizers by layer
# ONNX node names typically contain layer index info
# Let's check the naming pattern
print("\nSample param quantizer names:")
for n in param_quantizers[:10]:
    print(f"  {n}")
print("...")
for n in param_quantizers[-10:]:
    print(f"  {n}")

# Test: enable only GatedDeltaNet layers vs only full_attention layers
# First, let's see all unique prefix patterns
from collections import defaultdict
layer_groups = defaultdict(list)

for name in param_quantizers:
    # Try to extract layer index from name
    # Common patterns: "model.layers.0.xxx" or node names with numbers
    parts = name.split(".")
    assigned = False
    for i, part in enumerate(parts):
        if part == "layers" and i+1 < len(parts) and parts[i+1].isdigit():
            layer_idx = int(parts[i+1])
            layer_groups[f"layer_{layer_idx}"].append(name)
            assigned = True
            break
    if not assigned:
        # Try to find numbers in the name
        if "lm_head" in name or "embed_tokens" in name:
            layer_groups["lm_head_or_embed"].append(name)
        else:
            layer_groups["other"].append(name)

print(f"\nLayer groups found: {sorted(layer_groups.keys())}")
for k in sorted(layer_groups.keys()):
    print(f"  {k}: {len(layer_groups[k])} quantizers")

# Test 1: Enable ONLY the lm_head/embed quantizers
print("\n=== Test: Enable only lm_head/embed quantizers ===")
for n in layer_groups.get("lm_head_or_embed", []):
    quant_sim.qc_quantize_op_dict[n].enabled = True
quant_model._linear_attn_cache.clear()
out = generator(input_ids, torch.ones_like(input_ids))
cos = cosine(out.logits, logits_fp)
top_tok = tokenizer.decode([out.logits[0,-1].argmax()])
print(f"  Cosine vs FP: {cos:.6f}, Top: {top_tok!r}")
for n in layer_groups.get("lm_head_or_embed", []):
    quant_sim.qc_quantize_op_dict[n].enabled = False

# Test 2: Enable all param quantizers at once
print("\n=== Test: Enable ALL param quantizers ===")
for n in param_quantizers:
    quant_sim.qc_quantize_op_dict[n].enabled = True
quant_model._linear_attn_cache.clear()
out = generator(input_ids, torch.ones_like(input_ids))
cos = cosine(out.logits, logits_fp)
top_tok = tokenizer.decode([out.logits[0,-1].argmax()])
print(f"  Cosine vs FP: {cos:.6f}, Top: {top_tok!r}")
for n in param_quantizers:
    quant_sim.qc_quantize_op_dict[n].enabled = False

# Test 3: Enable param quantizers per-layer-group
print("\n=== Test: Enable quantizers per layer ===")
for layer_name in sorted(layer_groups.keys()):
    if layer_name == "other" and not layer_groups[layer_name]:
        continue
    for n in layer_groups[layer_name]:
        quant_sim.qc_quantize_op_dict[n].enabled = True
    quant_model._linear_attn_cache.clear()
    out = generator(input_ids, torch.ones_like(input_ids))
    cos = cosine(out.logits, logits_fp)
    top_tok = tokenizer.decode([out.logits[0,-1].argmax()])
    print(f"  {layer_name} ({len(layer_groups[layer_name])} qtzrs): cos={cos:.4f}, top={top_tok!r}")
    for n in layer_groups[layer_name]:
        quant_sim.qc_quantize_op_dict[n].enabled = False

# Test 4: Enable activation quantizers only
print("\n=== Test: Enable only activation quantizers (INT8) ===")
for n in act_quantizers:
    quant_sim.qc_quantize_op_dict[n].enabled = True
quant_model._linear_attn_cache.clear()
out = generator(input_ids, torch.ones_like(input_ids))
cos = cosine(out.logits, logits_fp)
top_tok = tokenizer.decode([out.logits[0,-1].argmax()])
print(f"  Cosine vs FP: {cos:.6f}, Top: {top_tok!r}")
for n in act_quantizers:
    quant_sim.qc_quantize_op_dict[n].enabled = False

print("\nDONE")
