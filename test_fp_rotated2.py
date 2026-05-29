"""
Quick FP inference test: compare rotated-weight ONNX vs original HF model.
If rotated model gives garbage, SpinQuant broke the model.
"""
import sys, os, math
sys.path.insert(0, '/workspace/ai-hub-models/src')
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained('/workspace/qwen3_5_0_8b_w8a16_spinquant')
text = "The capital of France is"
tokens = tokenizer.encode(text, return_tensors='pt')
print(f"Input: '{text}' -> tokens shape {tokens.shape}")

# Load HF model
print("Loading HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    torch_dtype=torch.float32,
    trust_remote_code=True
)
hf_model.eval()

# Get HF model predictions
print("Running HF inference...")
with torch.no_grad():
    hf_out = hf_model(tokens, use_cache=False)
    hf_logits = hf_out.logits[0, -1, :].float().numpy()  # logits for last token

hf_top5 = np.argsort(hf_logits)[-5:][::-1]
print(f"\nHF model top-5 predictions:")
for idx in hf_top5:
    print(f"  '{tokenizer.decode([idx])}' (logit={hf_logits[idx]:.2f})")

# Now get embeddings for ONNX model
seq_len = 128
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
padded = torch.full((1, seq_len), pad_id, dtype=torch.long)
padded[0, :tokens.shape[1]] = tokens[0]

with torch.no_grad():
    inputs_embeds = hf_model.model.embed_tokens(padded).float().numpy()

# Free HF model memory
del hf_model
import gc; gc.collect()

# Prepare ONNX inputs
context_length = 256
attn_mask = np.full((1, 1, seq_len, context_length), -50.0, dtype=np.float32)
for i in range(seq_len):
    attn_mask[0, 0, i, :i+1] = 0.0

# Position embeddings - need to match what the model expects
# The model uses rotary with head_dim=32 for position encoding
# Let me get the right frequencies from the HF model config
positions = np.arange(seq_len, dtype=np.float32)
# Model input expects [1, 1, 128, 32] for cos/sin
# From Qwen3.5 config: HEAD_DIM for position encoding
# The input shape shows 32, which is likely half of head_dim=64 for the attention heads
# or the full 32 for some specific rotary config

# Actually let me just use what the model expects based on input shapes
# position_ids_cos: [1, 1, 128, 32], position_ids_sin: [1, 1, 128, 32]
# This is likely the precomputed cos/sin for RoPE with dim=32 per head pair
# For Qwen3.5-0.8B: rope_theta=10000, head_dim for attn=256 but only first 64 dims use RoPE?
# Actually the shape 32 might mean 32 pairs = 64 RoPE dims

# Use standard RoPE computation with dim=64 (giving 32 pairs)
rope_dim = 64
theta = 1000000.0  # Qwen3.5 uses rope_theta=1000000
inv_freq = 1.0 / (theta ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))
pos_cos = np.cos(np.outer(positions, inv_freq)).reshape(1, 1, seq_len, 32).astype(np.float32)
pos_sin = np.sin(np.outer(positions, inv_freq)).reshape(1, 1, seq_len, 32).astype(np.float32)

feed_dict = {
    'inputs_embeds': inputs_embeds.astype(np.float32),
    'attention_mask': attn_mask,
    'position_ids_cos': pos_cos,
    'position_ids_sin': pos_sin,
}

# Add zero states
for i in range(24):
    if i in [3, 7, 11, 15, 19, 23]:
        feed_dict[f'past_key_{i}_in'] = np.zeros((2, 1, 256, 128), dtype=np.float32)
        feed_dict[f'past_value_{i}_in'] = np.zeros((2, 1, 128, 256), dtype=np.float32)
    else:
        feed_dict[f'conv_state_{i}_in'] = np.zeros((1, 6144, 3), dtype=np.float32)
        feed_dict[f'recurrent_state_{i}_in'] = np.zeros((1, 16, 128, 128), dtype=np.float32)

# Run ONNX inference
print("\nRunning ONNX inference on rotated-weight model...")
sess = ort.InferenceSession(
    '/workspace/qwen3_5_0_8b_w8a16_spinquant/model_seqlen128_cl256.onnx',
    providers=['CPUExecutionProvider']
)

# Verify all inputs present
model_inputs = {inp.name: inp.shape for inp in sess.get_inputs()}
for name in model_inputs:
    if name not in feed_dict:
        shape = [s if isinstance(s, int) else 1 for s in model_inputs[name]]
        feed_dict[name] = np.zeros(shape, dtype=np.float32)
        print(f"  Added zero for missing input: {name}")

outputs = sess.run(['logits'], feed_dict)
onnx_logits = outputs[0][0, tokens.shape[1]-1, :]  # logits at last real token position

print(f"\nONNX logits stats: min={onnx_logits.min():.2f}, max={onnx_logits.max():.2f}, std={onnx_logits.std():.4f}")
print(f"NaN: {np.isnan(onnx_logits).sum()}, Inf: {np.isinf(onnx_logits).sum()}")

onnx_top5 = np.argsort(onnx_logits)[-5:][::-1]
print(f"\nONNX rotated model top-5 predictions:")
for idx in onnx_top5:
    print(f"  '{tokenizer.decode([idx])}' (logit={onnx_logits[idx]:.2f})")

# Compute correlation between HF and ONNX logits (if shapes match)
if len(hf_logits) == len(onnx_logits):
    corr = np.corrcoef(hf_logits, onnx_logits)[0, 1]
    print(f"\nCorrelation between HF and ONNX logits: {corr:.6f}")
    mse = np.mean((hf_logits - onnx_logits)**2)
    print(f"MSE between HF and ONNX logits: {mse:.4f}")
else:
    print(f"\nShape mismatch: HF={len(hf_logits)}, ONNX={len(onnx_logits)}")

print("\nDone.")
