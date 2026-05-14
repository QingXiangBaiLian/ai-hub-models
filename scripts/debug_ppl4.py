"""
Diagnostic: Test with exact evaluation parameters (context=4096, seq=2048).
"""
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)

# Use first 4096 tokens (one full sample as the evaluator would)
sample_ids = tokens["input_ids"][:, :4096]
print(f"Sample shape: {sample_ids.shape}")

# Vanilla reference: process full 4096 tokens in one forward pass
print("\n=== Vanilla HF Model (4096 tokens, single forward) ===")
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
print(f"Vanilla PPL (4096 single): {vanilla_ppl:.2f}")

# Save logits for comparison
vanilla_logits_first_chunk = logits[0, :2048, :].clone()
vanilla_logits_second_chunk = logits[0, 2048:, :].clone()
del vanilla_model, logits
import gc; gc.collect()

# Adapted model: context=4096, seq=2048 (two chunks)
print("\n=== Adapted: context=4096, seq=2048 (two chunks) ===")
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B
from qai_hub_models.models._shared.llm.generator import LLM_Generator

model = Qwen3_5_0_8B.from_pretrained(sequence_length=2048, context_length=4096)
model.eval()
gen = LLM_Generator(models=[model], tokenizer=tokenizer, embedding=model.embedding)
gen.eval()
with torch.no_grad():
    out = gen(input_ids=sample_ids, attention_mask=torch.ones_like(sample_ids))
adapted_logits = out.logits
print(f"Adapted logits shape: {adapted_logits.shape}")

shift_logits = adapted_logits[..., :-1, :].contiguous().float()
shift_labels = sample_ids[..., 1:].contiguous()
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
adapted_ppl = math.exp(loss.item())
print(f"Adapted PPL (4096 chunked): {adapted_ppl:.2f}")

# Compare logits per chunk
print("\n=== Logit comparison ===")
adapted_first = adapted_logits[0, :2048, :]
adapted_second = adapted_logits[0, 2048:, :]

diff1 = (vanilla_logits_first_chunk - adapted_first).abs()
diff2 = (vanilla_logits_second_chunk - adapted_second).abs()
print(f"First chunk (tokens 0-2047):")
print(f"  Max diff: {diff1.max().item():.4f}, Mean diff: {diff1.mean().item():.4f}")
print(f"Second chunk (tokens 2048-4095):")
print(f"  Max diff: {diff2.max().item():.4f}, Mean diff: {diff2.mean().item():.4f}")

# Also compute per-chunk PPL
shift_logits_1 = adapted_logits[:, :2048, :][..., :-1, :].contiguous().float()
shift_labels_1 = sample_ids[:, :2048][..., 1:].contiguous()
loss1 = loss_fct(shift_logits_1.view(-1, shift_logits_1.size(-1)), shift_labels_1.view(-1))
print(f"\nPer-chunk PPL:")
print(f"  Chunk 1 PPL: {math.exp(loss1.item()):.2f}")

shift_logits_2 = adapted_logits[:, 2048:, :][..., :-1, :].contiguous().float()
shift_labels_2 = sample_ids[:, 2048:][..., 1:].contiguous()
loss2 = loss_fct(shift_logits_2.view(-1, shift_logits_2.size(-1)), shift_labels_2.view(-1))
print(f"  Chunk 2 PPL: {math.exp(loss2.item()):.2f}")

print(f"\n=== Summary ===")
print(f"Vanilla PPL: {vanilla_ppl:.2f}")
print(f"Adapted PPL: {adapted_ppl:.2f}")
print(f"Ratio: {adapted_ppl/vanilla_ppl:.2f}x")
