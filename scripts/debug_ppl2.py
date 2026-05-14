"""
Diagnostic: Compare vanilla HF PPL vs adapted model PPL on a WikiText sample.
"""
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

# Get a WikiText sample (first 512 tokens for quick test)
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
sample_ids = tokens["input_ids"][:, :512]  # First 512 tokens
print(f"Sample shape: {sample_ids.shape}")

# --- Test 1: Vanilla HF model PPL (no monkey patching) ---
print("\n=== Vanilla HF Model ===")
vanilla_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32
)
vanilla_model.eval()

with torch.no_grad():
    out = vanilla_model(sample_ids)
    logits = out.logits
    
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = sample_ids[..., 1:].contiguous()
loss_fct = CrossEntropyLoss()
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
vanilla_ppl = math.exp(loss.item())
print(f"Vanilla PPL: {vanilla_ppl:.2f}")
print(f"Loss: {loss.item():.4f}")

# Save vanilla logits for comparison
vanilla_logits_last = logits[0, -1, :].clone()

del vanilla_model
import gc; gc.collect()

# --- Test 2: Our adapted model through the generator ---
print("\n=== Adapted Model (via Generator) ===")
# This imports and monkey-patches
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B

model = Qwen3_5_0_8B.from_pretrained(
    sequence_length=512,  # Match the sample size
    context_length=512,   # No chunking needed
)
model.eval()

from qai_hub_models.models._shared.llm.generator import LLM_Generator

generator = LLM_Generator(
    models=[model],
    tokenizer=tokenizer,
    embedding=model.embedding,
)
generator.eval()

with torch.no_grad():
    attention_mask = torch.ones_like(sample_ids)
    out = generator(input_ids=sample_ids, attention_mask=attention_mask)
    adapted_logits = out.logits

print(f"Adapted logits shape: {adapted_logits.shape}")
shift_logits = adapted_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
shift_labels = sample_ids[..., 1:].contiguous()
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
adapted_ppl = math.exp(loss.item())
print(f"Adapted PPL: {adapted_ppl:.2f}")
print(f"Loss: {loss.item():.4f}")

# Compare logits at the last position
adapted_logits_last = adapted_logits[0, -1, :]
diff = (vanilla_logits_last - adapted_logits_last).abs()
print(f"\nLogit comparison (last position):")
print(f"  Max abs diff: {diff.max().item():.6f}")
print(f"  Mean abs diff: {diff.mean().item():.6f}")

print(f"\n=== Summary ===")
print(f"Vanilla PPL: {vanilla_ppl:.2f}")
print(f"Adapted PPL: {adapted_ppl:.2f}")
print(f"Ratio: {adapted_ppl/vanilla_ppl:.2f}x")
if adapted_ppl > vanilla_ppl * 1.5:
    print("WARNING: Adapted model has significantly higher PPL!")
