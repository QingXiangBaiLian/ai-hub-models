"""Diagnose INT8 KV cache quantization impact on Qwen3.5 model quality."""
import sys, os
sys.path.insert(0, "/workspace/ai-hub-models/src")
import numpy as np
import torch

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
print(f"Input: {text!r}, tokens: {input_ids.shape}")

EmbeddingClass = FP_Model.EmbeddingClass
embedding = EmbeddingClass(max_length=256, config=quant_model.llm_config)

# Run with full quantization
print("\n=== Full Quantization (W4+INT8 KV) ===")
generator = LLM_Generator([quant_model], tokenizer, embedding, accumulate_logits_on_cpu=True)
out_full = generator(input_ids, torch.ones_like(input_ids))
logits_full = out_full.logits
print(f"Logits stats: min={logits_full.min():.4f}, max={logits_full.max():.4f}, mean={logits_full.mean():.4f}, std={logits_full.std():.4f}")
top5 = torch.topk(logits_full[0, -1], 5)
for t, v in zip(top5.indices, top5.values):
    print(f"  {tokenizer.decode([t])!r}: {v.item():.3f}")

# Disable INT8 activation quantizers (KV cache quantization)
print("\n--- Disabling INT8 activation quantizers ---")
disabled_count = 0
disabled_names = []
quant_sim = quant_model.quant_sim
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names:
        if qc_op.enabled and qc_op.bitwidth == 8:
            qc_op.enabled = False
            disabled_count += 1
            disabled_names.append(name)
print(f"Disabled {disabled_count} INT8 activation quantizers")
if disabled_count <= 10:
    for n in disabled_names:
        print(f"  {n}")
else:
    print(f"  First 5: {disabled_names[:5]}")
    print(f"  Last 5: {disabled_names[-5:]}")

# Run without INT8 KV cache quantization
print("\n=== W4 only (no INT8 KV) ===")
# Clear linear attn cache to reset state
quant_model._linear_attn_cache.clear()
out_no_int8 = generator(input_ids, torch.ones_like(input_ids))
logits_no_int8 = out_no_int8.logits
print(f"Logits stats: min={logits_no_int8.min():.4f}, max={logits_no_int8.max():.4f}, mean={logits_no_int8.mean():.4f}, std={logits_no_int8.std():.4f}")
top5 = torch.topk(logits_no_int8[0, -1], 5)
for t, v in zip(top5.indices, top5.values):
    print(f"  {tokenizer.decode([t])!r}: {v.item():.3f}")

# Also disable all weight quantizers to get unquantized ONNX baseline
print("\n--- Disabling ALL quantizers ---")
disabled_w = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if qc_op.enabled:
        qc_op.enabled = False
        disabled_w += 1
print(f"Disabled {disabled_w} additional quantizers (all remaining)")

print("\n=== No Quantization (ONNX FP baseline) ===")
quant_model._linear_attn_cache.clear()
out_fp = generator(input_ids, torch.ones_like(input_ids))
logits_fp = out_fp.logits
print(f"Logits stats: min={logits_fp.min():.4f}, max={logits_fp.max():.4f}, mean={logits_fp.mean():.4f}, std={logits_fp.std():.4f}")
top5 = torch.topk(logits_fp[0, -1], 5)
for t, v in zip(top5.indices, top5.values):
    print(f"  {tokenizer.decode([t])!r}: {v.item():.3f}")

# Compare
print("\n=== Comparisons ===")
cos_full_vs_noint8 = torch.nn.functional.cosine_similarity(
    logits_full.flatten().unsqueeze(0), logits_no_int8.flatten().unsqueeze(0))
cos_full_vs_fp = torch.nn.functional.cosine_similarity(
    logits_full.flatten().unsqueeze(0), logits_fp.flatten().unsqueeze(0))
cos_noint8_vs_fp = torch.nn.functional.cosine_similarity(
    logits_no_int8.flatten().unsqueeze(0), logits_fp.flatten().unsqueeze(0))
print(f"Cosine(full_quant vs no_int8_kv): {cos_full_vs_noint8.item():.6f}")
print(f"Cosine(full_quant vs ONNX_fp): {cos_full_vs_fp.item():.6f}")
print(f"Cosine(no_int8_kv vs ONNX_fp): {cos_noint8_vs_fp.item():.6f}")
