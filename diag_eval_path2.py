import sys
sys.path.insert(0, "/workspace/ai-hub-models/src")
import torch
import numpy as np
import torch.nn.functional as F

from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.quantization_aimet_onnx import Precision

print("Loading model...")
model = Qwen3_5_0_8B_AIMETOnnx.from_pretrained(
    host_device=torch.device("cpu"),
    sequence_length=128,
    context_length=256,
    precision=Precision.w4,
    checkpoint="/workspace/qwen3_5_0_8b_w4a16_static",
)
print(f"Model loaded. llm_io_type={model.llm_io_type}")
print(f"attention_mask_min_clip={model.attention_mask_min_clip}")
print(f"sequence_length={model.sequence_length}, context_length={model.context_length}")

# Check embedding table
embed_table = model._get_embedding_table()
print(f"Embedding table: num_embeddings={embed_table.num_embeddings}, embedding_dim={embed_table.embedding_dim}")
print(f"Embedding weight shape: {embed_table.weight.shape}")

# Create test input
tokenizer = model.tokenizer
input_text = "The capital of France is Paris. The Eiffel Tower is a famous landmark. " * 5
tokens = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
input_ids = tokens["input_ids"][:, :128]
if input_ids.shape[1] < 128:
    pad_len = 128 - input_ids.shape[1]
    pad_id = tokenizer.eos_token_id or 0
    input_ids = torch.cat([torch.full((1, pad_len), pad_id, dtype=input_ids.dtype), input_ids], dim=1)
attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
print(f"\nTest input: shape={input_ids.shape}")
print(f"First 5 tokens: {input_ids[0,:5].tolist()}")

# Setup Generator properly
from qai_hub_models.models._shared.llm.generator import LLM_Generator
from transformers import DynamicCache

embedding = model.embedding
generator = LLM_Generator(
    [model],
    model.tokenizer,
    embedding,
    accumulate_logits_on_cpu=True,
)

# Method 1: Generator path
print("\n--- Generator path ---")
model._linear_attn_cache.clear()
with torch.no_grad():
    gen_result = generator(input_ids, attention_mask, DynamicCache())
gen_logits = gen_result.logits
print(f"Logits shape: {gen_logits.shape}")
print(f"Logits stats: min={gen_logits.min():.4f}, max={gen_logits.max():.4f}, std={gen_logits.std():.4f}")
print(f"Any NaN: {torch.isnan(gen_logits).any()}, Any Inf: {torch.isinf(gen_logits).any()}")

# Compute PPL from generator
shift_logits = gen_logits[0, :-1, :]
shift_labels = input_ids[0, 1:]
loss = F.cross_entropy(shift_logits, shift_labels)
ppl = torch.exp(loss)
print(f"PPL (generator): {ppl.item():.2f}")

# Method 2: Direct session
print("\n--- Direct session path ---")
model._linear_attn_cache.clear()

input_embeds = embed_table(input_ids)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

seq_len = 128
ctx_len = 256
kv_pad_len = ctx_len - seq_len

padded_attn = torch.cat([torch.zeros(1, kv_pad_len), attention_mask], dim=-1)
position_ids = torch.cumsum(padded_attn, dim=1, dtype=torch.int32) - 1
position_ids = position_ids.clip(0, ctx_len - 1)[..., -seq_len:]

attn_converter = AttentionMaskConverter(True)
cm_mask = attn_converter.to_4d(padded_attn, query_length=seq_len, key_value_length=ctx_len, dtype=torch.float32)
cm_mask = cm_mask.clip(min=model.attention_mask_min_clip)
cm_mask = model.attention_mask_multiplier * cm_mask

pos_cos, pos_sin = embedding.get_embedding(position_ids)

session = model.quant_sim.session
onnx_input_names = [inp.name for inp in session.get_inputs()]

input_feed = {}
input_feed["inputs_embeds"] = input_embeds.detach().numpy()
input_feed["attention_mask"] = cm_mask.detach().numpy()
input_feed["position_ids_cos"] = pos_cos.detach().numpy()
input_feed["position_ids_sin"] = pos_sin.detach().numpy()

for name in onnx_input_names:
    if name not in input_feed:
        inp_info = next(inp for inp in session.get_inputs() if inp.name == name)
        shape = [d if isinstance(d, int) else 1 for d in inp_info.shape]
        input_feed[name] = np.zeros(shape, dtype=np.float32)

output_names = [out.name for out in session.get_outputs()]
output_np = session.run(None, input_feed)
output_dict = dict(zip(output_names, output_np))

logits_key = "logits" if "logits" in output_dict else next(k for k in output_dict if "logit" in k.lower())
direct_logits = torch.from_numpy(output_dict[logits_key])
print(f"Direct logits shape: {direct_logits.shape}")
print(f"Direct logits stats: min={direct_logits.min():.4f}, max={direct_logits.max():.4f}, std={direct_logits.std():.4f}")

# PPL from direct
shift_logits_d = direct_logits[0, :-1, :]
loss_d = F.cross_entropy(shift_logits_d, shift_labels)
ppl_d = torch.exp(loss_d)
print(f"PPL (direct): {ppl_d.item():.2f}")

# Compare
if gen_logits.shape == direct_logits.shape:
    diff = (gen_logits - direct_logits).abs()
    print(f"\n--- Comparison ---")
    print(f"Max abs diff: {diff.max():.6f}")
    print(f"Mean abs diff: {diff.mean():.6f}")
else:
    print(f"Shape mismatch: gen={gen_logits.shape}, direct={direct_logits.shape}")

print("\nDone.")
