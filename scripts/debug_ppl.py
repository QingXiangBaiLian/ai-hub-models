"""
Diagnostic: Compare logits between vanilla HF Qwen3.5-0.8B and our adapted model.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Short text for comparison
text = "The capital of France is"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
input_ids = tokenizer(text, return_tensors="pt").input_ids
print(f"Input shape: {input_ids.shape}")
print(f"Tokens: {input_ids}")

# --- Vanilla HF model (no monkey patching) ---
print("\n=== Vanilla HF model ===")
vanilla_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    torch_dtype=torch.float32,
)
vanilla_model.eval()
with torch.no_grad():
    vanilla_out = vanilla_model(input_ids)
    vanilla_logits = vanilla_out.logits
print(f"Vanilla logits shape: {vanilla_logits.shape}")
print(f"Vanilla logits (last token, top 5):")
top5 = torch.topk(vanilla_logits[0, -1], 5)
for idx, val in zip(top5.indices, top5.values):
    print(f"  {tokenizer.decode([idx])!r}: {val.item():.4f}")
print(f"Vanilla logits stats: mean={vanilla_logits.mean():.4f}, std={vanilla_logits.std():.4f}")

del vanilla_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# --- Our adapted model ---
print("\n=== Adapted model (with monkey patching) ===")
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B
model = Qwen3_5_0_8B.from_pretrained()
model.eval()

# Call the underlying HF model directly (after monkey patching)
# This bypasses the generator/SHA framework
with torch.no_grad():
    adapted_out = model.model(input_ids)
    adapted_logits = adapted_out.logits

print(f"Adapted logits shape: {adapted_logits.shape}")
print(f"Adapted logits (last token, top 5):")
top5 = torch.topk(adapted_logits[0, -1], 5)
for idx, val in zip(top5.indices, top5.values):
    print(f"  {tokenizer.decode([idx])!r}: {val.item():.4f}")
print(f"Adapted logits stats: mean={adapted_logits.mean():.4f}, std={adapted_logits.std():.4f}")

# Compare
print("\n=== Comparison ===")
# Align shapes - adapted might have padding
min_len = min(vanilla_logits.shape[1], adapted_logits.shape[1])
v_logits = vanilla_logits[0, :min_len]
a_logits = adapted_logits[0, -min_len:]  # last tokens (due to padding)
diff = (v_logits - a_logits).abs()
print(f"Max abs diff: {diff.max().item():.6f}")
print(f"Mean abs diff: {diff.mean().item():.6f}")
if diff.max().item() > 0.01:
    print("WARNING: Significant logit mismatch detected!")
    # Find which position has max diff
    max_pos = diff.max(dim=-1).values.argmax().item()
    print(f"  Max diff at position {max_pos}")
    print(f"  Vanilla top token: {tokenizer.decode([v_logits[max_pos].argmax()])!r}")
    print(f"  Adapted top token: {tokenizer.decode([a_logits[max_pos].argmax()])!r}")
else:
    print("OK: Logits match closely.")
