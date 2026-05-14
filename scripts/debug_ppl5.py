"""
Diagnostic: Test that _linear_attn_cache leaks between samples.
"""
import torch
import math
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(dataset["text"])
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)

from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B
from qai_hub_models.models._shared.llm.generator import LLM_Generator

model = Qwen3_5_0_8B.from_pretrained(sequence_length=2048, context_length=4096)
model.eval()
embedding = model.embedding
gen = LLM_Generator(models=[model], tokenizer=tokenizer, embedding=embedding, accumulate_logits_on_cpu=True)
gen.eval()

loss_fct = CrossEntropyLoss()
losses = []

# Process first 5 samples WITHOUT resetting cache between them
print("=== Without cache reset (current behavior) ===")
for i in range(5):
    sample_ids = tokens["input_ids"][:, i*4096:(i+1)*4096]
    if sample_ids.shape[1] < 4096:
        break
    with torch.no_grad():
        out = gen(input_ids=sample_ids, attention_mask=torch.ones_like(sample_ids))
    shift_logits = out.logits[..., :-1, :].contiguous().float()
    shift_labels = sample_ids[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
    ppl = math.exp(loss)
    has_cache = hasattr(model, "_linear_attn_cache") and len(model._linear_attn_cache) > 0
    print(f"  Sample {i}: PPL={ppl:.2f}, loss={loss:.4f}, has_linear_cache={has_cache}")
    losses.append(loss)

avg_loss = sum(losses) / len(losses)
print(f"  Average PPL (no reset): {math.exp(avg_loss):.2f}")

# Now reset cache between samples
print("\n=== With cache reset (correct behavior) ===")
losses2 = []
for i in range(5):
    # RESET the linear attention cache before each sample
    if hasattr(model, "_linear_attn_cache"):
        del model._linear_attn_cache
    
    sample_ids = tokens["input_ids"][:, i*4096:(i+1)*4096]
    if sample_ids.shape[1] < 4096:
        break
    with torch.no_grad():
        out = gen(input_ids=sample_ids, attention_mask=torch.ones_like(sample_ids))
    shift_logits = out.logits[..., :-1, :].contiguous().float()
    shift_labels = sample_ids[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
    ppl = math.exp(loss)
    print(f"  Sample {i}: PPL={ppl:.2f}, loss={loss:.4f}")
    losses2.append(loss)

avg_loss2 = sum(losses2) / len(losses2)
print(f"  Average PPL (with reset): {math.exp(avg_loss2):.2f}")

print(f"\n=== Summary ===")
print(f"No reset: {math.exp(avg_loss):.2f}")
print(f"With reset: {math.exp(avg_loss2):.2f}")
