"""Generate text using only the seq_len=128 model (avoids seq_len=1 encoding mismatch)."""
import sys
import os
sys.path.insert(0, '/workspace/ai-hub-models/src')
os.environ['PYTHONUNBUFFERED'] = '1'

CHECKPOINT = '/workspace/qwen3_5_0_8b_w4a16_static'
PROMPT = "你是谁呀？"
MAX_NEW_TOKENS = 100

print(f"Loading model from {CHECKPOINT}")
print(f"Prompt: {PROMPT}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print()

import torch
from transformers import AutoTokenizer, GenerationConfig

from qai_hub_models.models._shared.llm.generator import LLM_Generator, LLM_Loader
from qai_hub_models.models._shared.qwen3_5.model import END_TOKENS, Qwen3_5RopeEmbedding
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.models._shared.llm.model import get_llm_config

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print(f"Tokenizer loaded.")

host_device = torch.device("cpu")
context_length = 256

# Only use seq_len=128 (avoid seq_len=1 encoding mismatch)
model_params = {
    "context_length": context_length,
    "host_device": host_device,
    "checkpoint": CHECKPOINT,
}

# Create a single loader for seq_len=128 only
models = [
    LLM_Loader(Qwen3_5_0_8B_AIMETOnnx, 128, model_params, host_device),
]

config = get_llm_config(CHECKPOINT)

# Create rope embedding
rope_embedding = Qwen3_5RopeEmbedding(
    max_length=context_length, config=config
)

# Create the generator
inferencer = LLM_Generator(
    models,
    tokenizer,
    rope_embedding,
)

# Set generation config
end_token_ids = [tokenizer.eos_token_id]
for token in END_TOKENS:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) == 1:
        end_token_ids.append(token_ids[0])

inferencer.generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    eos_token_id=end_token_ids,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,  # greedy for reproducibility
)

# Prepare prompt with chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": PROMPT},
]
input_prompt_processed = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(f"Formatted prompt:\n{input_prompt_processed}\n")

input_tokens = tokenizer(
    input_prompt_processed,
    return_tensors="pt",
    add_special_tokens=False,
).to(host_device)

print(f"Input tokens: {input_tokens['input_ids'].shape[1]}")
print(f"\nGenerating (greedy, max {MAX_NEW_TOKENS} tokens)...")
print("=" * 60)

# Generate
from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=False)

output = inferencer.generate(
    inputs=input_tokens["input_ids"],
    attention_mask=input_tokens["attention_mask"],
    generation_config=inferencer.generation_config,
    streamer=streamer,
)

print("\n" + "=" * 60)
print("Generation complete!")

# Decode full output
full_text = tokenizer.decode(output[0], skip_special_tokens=False)
print(f"\nFull output (with special tokens):\n{full_text}")

inferencer.cleanup()
