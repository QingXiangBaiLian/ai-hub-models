"""
Diagnostic: Test PPL with chunked processing (context_length > sequence_length).
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
sample_ids = tokens["input_ids"][:, :1024]  # 1024 tokens
print(f"Sample shape: {sample_ids.shape}")

# Vanilla reference
print("\n=== Vanilla HF Model (1024 tokens, single forward) ===")
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
print(f"Vanilla PPL (1024 single): {vanilla_ppl:.2f}")
del vanilla_model
import gc; gc.collect()

# Adapted model: SINGLE CHUNK (context=1024, seq=1024)
print("\n=== Adapted: context=1024, seq=1024 (single chunk) ===")
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B
from qai_hub_models.models._shared.llm.generator import LLM_Generator

model_single = Qwen3_5_0_8B.from_pretrained(sequence_length=1024, context_length=1024)
model_single.eval()
gen_single = LLM_Generator(models=[model_single], tokenizer=tokenizer, embedding=model_single.embedding)
gen_single.eval()
with torch.no_grad():
    out = gen_single(input_ids=sample_ids, attention_mask=torch.ones_like(sample_ids))
shift_logits = out.logits[..., :-1, :].contiguous().float()
shift_labels = sample_ids[..., 1:].contiguous()
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
single_ppl = math.exp(loss.item())
print(f"Single chunk PPL: {single_ppl:.2f}")
del model_single, gen_single; gc.collect()

# Adapted model: TWO CHUNKS (context=1024, seq=512)
print("\n=== Adapted: context=1024, seq=512 (two chunks) ===")
model_chunked = Qwen3_5_0_8B.from_pretrained(sequence_length=512, context_length=1024)
model_chunked.eval()
gen_chunked = LLM_Generator(models=[model_chunked], tokenizer=tokenizer, embedding=model_chunked.embedding)
gen_chunked.eval()
with torch.no_grad():
    out = gen_chunked(input_ids=sample_ids, attention_mask=torch.ones_like(sample_ids))
adapted_logits = out.logits
print(f"Chunked logits shape: {adapted_logits.shape}")
shift_logits = adapted_logits[..., :-1, :].contiguous().float()
shift_labels = sample_ids[..., 1:].contiguous()
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
chunked_ppl = math.exp(loss.item())
print(f"Two-chunk PPL: {chunked_ppl:.2f}")

print(f"\n=== Summary ===")
print(f"Vanilla (full):   {vanilla_ppl:.2f}")
print(f"Adapted (single): {single_ppl:.2f}")
print(f"Adapted (chunked):{chunked_ppl:.2f}")
if chunked_ppl > single_ppl * 1.5:
    print("BUG: Chunked processing degrades PPL significantly!")
