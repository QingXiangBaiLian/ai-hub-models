import sys, os
sys.path.insert(0, "/workspace/ai-hub-models/src")
import numpy as np
import torch

from qai_hub_models.models.qwen3_5_0_8b import FP_Model, Model
from qai_hub_models.models._shared.llm.generator import LLM_Generator

print("Loading FP model...")
fp_model = FP_Model.from_pretrained(sequence_length=128, context_length=256).to("cpu")

print("Loading quantized model...")
quant_model = Model.from_pretrained(
    checkpoint="/workspace/qwen3_5_0_8b_w4a16_static",
    sequence_length=128,
    context_length=256,
    host_device=torch.device("cpu"),
).to("cpu")

tokenizer = fp_model.tokenizer
text = "The capital of France is"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
print(f"Input: {text!r}, tokens: {input_ids.shape}")

EmbeddingClass = FP_Model.EmbeddingClass
embedding = EmbeddingClass(max_length=256, config=fp_model.llm_config)

print("\n=== FP Model ===")
fp_generator = LLM_Generator([fp_model], tokenizer, embedding, accumulate_logits_on_cpu=True)
fp_out = fp_generator(input_ids, torch.ones_like(input_ids))
fp_logits = fp_out.logits
print(f"FP logits shape: {fp_logits.shape}")
print(f"FP logits stats: min={fp_logits.min():.4f}, max={fp_logits.max():.4f}, mean={fp_logits.mean():.4f}, std={fp_logits.std():.4f}")
fp_top5 = torch.topk(fp_logits[0, -1], 5)
for t, v in zip(fp_top5.indices, fp_top5.values):
    print(f"  {tokenizer.decode([t])!r}: {v.item():.3f}")

del fp_model, fp_generator
import gc; gc.collect()

print("\n=== Quantized Model ===")
quant_generator = LLM_Generator([quant_model], tokenizer, embedding, accumulate_logits_on_cpu=True)
quant_out = quant_generator(input_ids, torch.ones_like(input_ids))
quant_logits = quant_out.logits
print(f"Quant logits shape: {quant_logits.shape}")
print(f"Quant logits stats: min={quant_logits.min():.4f}, max={quant_logits.max():.4f}, mean={quant_logits.mean():.4f}, std={quant_logits.std():.4f}")
quant_top5 = torch.topk(quant_logits[0, -1], 5)
for t, v in zip(quant_top5.indices, quant_top5.values):
    print(f"  {tokenizer.decode([t])!r}: {v.item():.3f}")

print("\n=== Comparison ===")
cos_sim = torch.nn.functional.cosine_similarity(fp_logits.flatten().unsqueeze(0), quant_logits.flatten().unsqueeze(0))
print(f"Cosine similarity: {cos_sim.item():.6f}")
print(f"Max abs diff: {(fp_logits - quant_logits).abs().max():.4f}")
print(f"FP logits norm: {fp_logits.norm():.4f}")
print(f"Quant logits norm: {quant_logits.norm():.4f}")
