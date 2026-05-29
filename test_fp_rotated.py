"""
Quick FP inference test on the rotated-weight ONNX model.
Compare logits quality to determine if SpinQuant rotations corrupted the model.
"""
import sys, os, math
sys.path.insert(0, '/workspace/ai-hub-models/src')

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('/workspace/qwen3_5_0_8b_w8a16_spinquant')

# Encode test text
text = "The capital of France is"
tokens = tokenizer.encode(text, return_tensors='pt')
print(f"Input tokens: {tokens.shape} = {tokens}")

# We need embeddings - load the embedding layer from the original model
print("Loading original HF model for embedding layer...")
from qai_hub_models.models.qwen3_5_0_8b import Model as Qwen3Model
hf_model = AutoModelForCausalLM.from_pretrained(
    Qwen3Model.get_hub_or_disk_model_id(),
    torch_dtype=torch.float32,
    trust_remote_code=True
)
embed_layer = hf_model.model.embed_tokens

# Get embeddings for first 128 tokens (pad if needed)
seq_len = 128
if tokens.shape[1] < seq_len:
    # Pad with pad_token_id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    padded = torch.full((1, seq_len), pad_id, dtype=torch.long)
    padded[0, :tokens.shape[1]] = tokens[0]
    tokens_padded = padded
else:
    tokens_padded = tokens[:, :seq_len]

with torch.no_grad():
    inputs_embeds = embed_layer(tokens_padded).numpy()

print(f"inputs_embeds shape: {inputs_embeds.shape}, dtype: {inputs_embeds.dtype}")
print(f"inputs_embeds stats: min={inputs_embeds.min():.4f}, max={inputs_embeds.max():.4f}, mean={inputs_embeds.mean():.4f}")

# Prepare other inputs for the ONNX model
context_length = 256

# Attention mask: causal mask for seq_len=128 within context_length=256
# Shape: [1, 1, 128, 256] - the first 128 positions can attend to themselves causally
attn_mask = np.full((1, 1, seq_len, context_length), -50.0, dtype=np.float32)
for i in range(seq_len):
    attn_mask[0, 0, i, :i+1] = 0.0  # can attend to positions 0..i

# Position IDs (cos/sin for rotary) - shape [1, 1, 128, 32]
# Use simple sequential positions
positions = np.arange(seq_len, dtype=np.float32)
head_dim = 32  # from model input shape
theta = 10000.0
freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
pos_cos = np.cos(np.outer(positions, freqs)).reshape(1, 1, seq_len, head_dim // 2)
pos_sin = np.sin(np.outer(positions, freqs)).reshape(1, 1, seq_len, head_dim // 2)

# Hmm, the model expects shape [1, 1, 128, 32] for both cos and sin
# The dim is 32, so there are 32 frequency components (not 16)
freqs32 = 1.0 / (theta ** (np.arange(0, 64, 2, dtype=np.float32) / 64))
pos_cos = np.cos(np.outer(positions, freqs32)).reshape(1, 1, seq_len, 32).astype(np.float32)
pos_sin = np.sin(np.outer(positions, freqs32)).reshape(1, 1, seq_len, 32).astype(np.float32)

# Initialize all states to zeros
feed_dict = {
    'inputs_embeds': inputs_embeds.astype(np.float32),
    'attention_mask': attn_mask,
    'position_ids_cos': pos_cos,
    'position_ids_sin': pos_sin,
}

# Add zero-initialized conv states, recurrent states, and KV caches
# Based on model inputs we saw:
# conv_state_X_in: [1, 6144, 3]
# recurrent_state_X_in: [1, 16, 128, 128]  
# past_key_X_in: [2, 1, 256, 128]
# past_value_X_in: [2, 1, 128, 256]

for i in range(24):
    if i in [3, 7, 11, 15, 19, 23]:  # full_attention layers
        feed_dict[f'past_key_{i}_in'] = np.zeros((2, 1, 256, 128), dtype=np.float32)
        feed_dict[f'past_value_{i}_in'] = np.zeros((2, 1, 128, 256), dtype=np.float32)
    else:  # GatedDeltaNet layers
        feed_dict[f'conv_state_{i}_in'] = np.zeros((1, 6144, 3), dtype=np.float32)
        feed_dict[f'recurrent_state_{i}_in'] = np.zeros((1, 16, 128, 128), dtype=np.float32)

# Run inference on rotated-weight model
print("\nRunning FP inference on rotated-weight model...")
sess = ort.InferenceSession(
    '/workspace/qwen3_5_0_8b_w8a16_spinquant/model_seqlen128_cl256.onnx',
    providers=['CPUExecutionProvider']
)

# Check all expected inputs are provided
model_inputs = [inp.name for inp in sess.get_inputs()]
missing = [name for name in model_inputs if name not in feed_dict]
if missing:
    print(f"WARNING: Missing inputs: {missing}")
    # Add them as zeros with correct shapes
    for inp in sess.get_inputs():
        if inp.name not in feed_dict:
            shape = [s if isinstance(s, int) else 1 for s in inp.shape]
            feed_dict[inp.name] = np.zeros(shape, dtype=np.float32)
            print(f"  Added zero input for: {inp.name} shape={shape}")

outputs = sess.run(None, feed_dict)
logits = outputs[0]  # [1, 128, 248320]

print(f"\nLogits shape: {logits.shape}")
print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.6f}, std={logits.std():.4f}")

# Check logits for position 4 (after "The capital of France is") 
# Compare with what we'd expect - should have high prob for "Paris"
pos = tokens.shape[1] - 1  # last real token position
next_token_logits = logits[0, pos, :]
top_indices = np.argsort(next_token_logits)[-10:][::-1]
print(f"\nTop 10 predictions after '{text}':")
for idx in top_indices:
    token_str = tokenizer.decode([idx])
    print(f"  Token {idx}: '{token_str}' (logit={next_token_logits[idx]:.4f})")

# Check if logits contain NaN or Inf
print(f"\nNaN count: {np.isnan(logits).sum()}")
print(f"Inf count: {np.isinf(logits).sum()}")

# Compare with original model
print("\n\nRunning inference on ORIGINAL HF model for comparison...")
with torch.no_grad():
    hf_outputs = hf_model(tokens, use_cache=False)
    hf_logits = hf_outputs.logits.numpy()

print(f"HF logits shape: {hf_logits.shape}")
hf_next_logits = hf_logits[0, -1, :]
hf_top_indices = np.argsort(hf_next_logits)[-10:][::-1]
print(f"\nHF Top 10 predictions after '{text}':")
for idx in hf_top_indices:
    token_str = tokenizer.decode([idx])
    print(f"  Token {idx}: '{token_str}' (logit={hf_next_logits[idx]:.4f})")

del hf_model
print("\nDone.")
