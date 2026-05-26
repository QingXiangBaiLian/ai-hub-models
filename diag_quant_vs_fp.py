"""Test: Is the issue in ONNX export or quantization?
Run the model with remove_quantization() to test FP32 ONNX path.
"""
import torch
import math
from torch.nn import CrossEntropyLoss
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

SEQ_LEN = 128
CTX_LEN = 256

# Wikitext first 128 tokens (from the dataset)
print("Loading FP model to get tokenizer and wikitext tokens...")
from qai_hub_models.models.qwen3_5_0_8b.model import (
    Qwen3_5_0_8B,
    Qwen3_5_0_8B_AIMETOnnx,
)

fp_model = Qwen3_5_0_8B.from_pretrained(sequence_length=SEQ_LEN, context_length=CTX_LEN)

# Get wikitext tokens
from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets import get_dataset_from_name
dataset = get_dataset_from_name(
    name="wikitext", tokenizer=fp_model.tokenizer,
    block_size=SEQ_LEN, context_length=CTX_LEN,
    num_samples=1, split=DatasetSplit.TEST,
)
sample = dataset[0]
# First 128 tokens of wikitext
input_ids_128 = sample["input_ids"][:, :128]  # shape [1, 128]
attention_mask_128 = sample["attention_mask"][:, :128]  # shape [1, 128]
print(f"Wikitext first 128 tokens: shape={input_ids_128.shape}")
print(f"First 10 tokens: {input_ids_128[0, :10].tolist()}")
print(f"Attention mask all ones: {attention_mask_128.sum().item() == 128}")

# Also compute FP model PPL on same tokens for comparison
print("\nComputing FP model PPL on these tokens...")
fp_model.eval()
with torch.no_grad():
    # Run FP model using its native forward
    from qai_hub_models.models._shared.qwen3_5.model import Qwen3_5RopeEmbedding
    from qai_hub_models.models._shared.llm.generator import LLM_Generator
    
    embedding_rope = Qwen3_5RopeEmbedding(max_length=CTX_LEN, config=fp_model.llm_config)
    fp_generator = LLM_Generator([fp_model], fp_model.tokenizer, embedding_rope, accumulate_logits_on_cpu=True)
    fp_outputs = fp_generator(input_ids_128, attention_mask_128, None)
    fp_logits = fp_outputs.logits
    
    shift_logits = fp_logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids_128[:, 1:].contiguous()
    fp_loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
    print(f"FP model: loss={fp_loss:.4f}, PPL={math.exp(fp_loss):.2f}")

del fp_model, fp_generator
import gc
gc.collect()

# Load quantized model
print("\nLoading quantized model...")
model = Qwen3_5_0_8B_AIMETOnnx.from_pretrained(
    host_device=torch.device("cpu"),
    sequence_length=SEQ_LEN, context_length=CTX_LEN,
    precision="w4a16",
    checkpoint="/workspace/qwen3_5_0_8b_w4a16_static",
)
model.eval()

# Create generator
embedding_rope = Qwen3_5RopeEmbedding(max_length=CTX_LEN, config=model.llm_config)
generator = LLM_Generator([model], model.tokenizer, embedding_rope, accumulate_logits_on_cpu=True)

# Test 1: Quantized model (normal)
print("\n--- Test 1: Quantized model (normal) ---")
with torch.no_grad():
    outputs = generator(input_ids_128, attention_mask_128, None)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids_128[:, 1:].contiguous()
    loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
    print(f"Quantized: loss={loss:.4f}, PPL={math.exp(loss):.2f}")
    print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, std={logits.std():.4f}")

# Test 2: Model with remove_quantization (FP32 ONNX path)
print("\n--- Test 2: Model with remove_quantization ---")
model._linear_attn_cache.clear()
with torch.no_grad():
    with model.remove_quantization():
        outputs2 = generator(input_ids_128, attention_mask_128, None)
        logits2 = outputs2.logits
        shift_logits2 = logits2[:, :-1, :].contiguous().float()
        loss2 = CrossEntropyLoss()(shift_logits2.view(-1, shift_logits2.size(-1)), shift_labels.view(-1)).item()
        print(f"FP32 ONNX: loss={loss2:.4f}, PPL={math.exp(loss2):.2f}")
        print(f"Logits stats: min={logits2.min():.4f}, max={logits2.max():.4f}, std={logits2.std():.4f}")

# Test 3: Compare logits
print("\n--- Comparison ---")
diff = (logits - logits2).abs()
print(f"Quantized vs FP32 ONNX logits max diff: {diff.max():.4f}")
print(f"Quantized vs FP32 ONNX logits mean diff: {diff.mean():.4f}")

# Test 4: Check what tokens the model predicts at each position
print("\n--- Top-1 predictions (first 20 positions) ---")
pred_quantized = logits[0, :20, :].argmax(dim=-1)
pred_fp32 = logits2[0, :20, :].argmax(dim=-1)
actual_next = input_ids_128[0, 1:21]
print(f"Actual next tokens: {actual_next.tolist()}")
print(f"Quantized preds:    {pred_quantized.tolist()}")
print(f"FP32 ONNX preds:    {pred_fp32.tolist()}")
matches_quant = (pred_quantized == actual_next).sum().item()
matches_fp32 = (pred_fp32 == actual_next).sum().item()
print(f"Quantized matches: {matches_quant}/20")
print(f"FP32 ONNX matches: {matches_fp32}/20")
