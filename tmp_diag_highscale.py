"""Test: disable only the high-scale activation quantizers and measure improvement."""
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

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

# Test 1: Full quant baseline
print("\n=== Full quant (all quantizers enabled) ===")
quant_model._linear_attn_cache.clear()
out_full = generator(input_ids, torch.ones_like(input_ids))
logits_full = out_full.logits.clone()
top_full = tokenizer.decode([logits_full[0,-1].argmax()])
print(f"  Top token: {top_full!r}")

# Get FP baseline
print("\n=== FP baseline (all quantizers disabled) ===")
all_enabled = []
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if qc_op.enabled:
        all_enabled.append(name)
        qc_op.enabled = False
quant_model._linear_attn_cache.clear()
out_fp = generator(input_ids, torch.ones_like(input_ids))
logits_fp = out_fp.logits.clone()
top_fp = tokenizer.decode([logits_fp[0,-1].argmax()])
print(f"  Top token: {top_fp!r}")

cos_full_vs_fp = cosine(logits_full, logits_fp)
print(f"\n  Full quant vs FP cosine: {cos_full_vs_fp:.6f}")

# Re-enable all
for name in all_enabled:
    quant_sim.qc_quantize_op_dict[name].enabled = True

# Identify high-scale activation quantizers
# These are the ones with scale > 0.01 based on encoding analysis
high_scale_names = set()
import json
with open("/workspace/qwen3_5_0_8b_w4a16_static/model.encodings") as f:
    data = json.load(f)
for enc in data['activation_encodings']:
    if enc['bw'] == 16 and 'scale' in enc and max(enc['scale']) > 0.01:
        high_scale_names.add(enc['name'])

print(f"\nHigh-scale activation quantizers (scale > 0.01): {len(high_scale_names)}")

# Test 2: Disable only high-scale activation quantizers
print("\n=== Full quant MINUS high-scale activations (scale > 0.01) ===")
disabled_high = []
for name in high_scale_names:
    if name in quant_sim.qc_quantize_op_dict:
        qc_op = quant_sim.qc_quantize_op_dict[name]
        if qc_op.enabled:
            qc_op.enabled = False
            disabled_high.append(name)
print(f"  Disabled {len(disabled_high)} quantizers")
quant_model._linear_attn_cache.clear()
out_nohigh = generator(input_ids, torch.ones_like(input_ids))
cos_nohigh = cosine(out_nohigh.logits, logits_fp)
top_nohigh = tokenizer.decode([out_nohigh.logits[0,-1].argmax()])
print(f"  Cosine vs FP: {cos_nohigh:.6f}, Top: {top_nohigh!r}")
# Re-enable
for name in disabled_high:
    quant_sim.qc_quantize_op_dict[name].enabled = True

# Test 3: Disable only the top-50 (scale > 0.05)
print("\n=== Full quant MINUS top-50 activations (scale > 0.05) ===")
high_scale_50 = set()
for enc in data['activation_encodings']:
    if enc['bw'] == 16 and 'scale' in enc and max(enc['scale']) > 0.05:
        high_scale_50.add(enc['name'])
disabled_50 = []
for name in high_scale_50:
    if name in quant_sim.qc_quantize_op_dict:
        qc_op = quant_sim.qc_quantize_op_dict[name]
        if qc_op.enabled:
            qc_op.enabled = False
            disabled_50.append(name)
print(f"  Disabled {len(disabled_50)} quantizers")
quant_model._linear_attn_cache.clear()
out_no50 = generator(input_ids, torch.ones_like(input_ids))
cos_no50 = cosine(out_no50.logits, logits_fp)
top_no50 = tokenizer.decode([out_no50.logits[0,-1].argmax()])
print(f"  Cosine vs FP: {cos_no50:.6f}, Top: {top_no50!r}")
# Re-enable
for name in disabled_50:
    quant_sim.qc_quantize_op_dict[name].enabled = True

# Test 4: Disable ALL 16-bit activation quantizers (keep INT8 + params)
print("\n=== Disable ALL INT16 activations (keep params + INT8 acts) ===")
disabled_16 = []
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names and qc_op.enabled and qc_op.bitwidth == 16:
        qc_op.enabled = False
        disabled_16.append(name)
print(f"  Disabled {len(disabled_16)} INT16 activation quantizers")
quant_model._linear_attn_cache.clear()
out_no16 = generator(input_ids, torch.ones_like(input_ids))
cos_no16 = cosine(out_no16.logits, logits_fp)
top_no16 = tokenizer.decode([out_no16.logits[0,-1].argmax()])
print(f"  Cosine vs FP: {cos_no16:.6f}, Top: {top_no16!r}")
# Re-enable
for name in disabled_16:
    quant_sim.qc_quantize_op_dict[name].enabled = True

# Test 5: Disable ALL activation quantizers (keep only params)
print("\n=== Disable ALL activations (params only) ===")
disabled_all_act = []
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names and qc_op.enabled:
        qc_op.enabled = False
        disabled_all_act.append(name)
print(f"  Disabled {len(disabled_all_act)} activation quantizers")
quant_model._linear_attn_cache.clear()
out_noact = generator(input_ids, torch.ones_like(input_ids))
cos_noact = cosine(out_noact.logits, logits_fp)
top_noact = tokenizer.decode([out_noact.logits[0,-1].argmax()])
print(f"  Cosine vs FP: {cos_noact:.6f}, Top: {top_noact!r}")

print("\n=== SUMMARY ===")
print(f"  FP baseline: top={top_fp!r}")
print(f"  Full quant:           cos={cos_full_vs_fp:.4f}, top={top_full!r}")
print(f"  No high-scale (>0.01): cos={cos_nohigh:.4f}, top={top_nohigh!r}")
print(f"  No top-50 (>0.05):    cos={cos_no50:.4f}, top={top_no50!r}")
print(f"  No INT16 acts:         cos={cos_no16:.4f}, top={top_no16!r}")
print(f"  No acts at all:       cos={cos_noact:.4f}, top={top_noact!r}")
print("\nDONE")
