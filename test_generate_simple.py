"""Simple text generation using only the seq_len=1 model for token-by-token decode."""
import sys
import os
import numpy as np
import time

sys.path.insert(0, '/workspace/ai-hub-models/src')
os.environ['PYTHONUNBUFFERED'] = '1'

CHECKPOINT = '/workspace/qwen3_5_0_8b_w4a16_static'
PROMPT = "你是谁呀？"
MAX_NEW_TOKENS = 50

print(f"Loading model from {CHECKPOINT}")
print(f"Prompt: {PROMPT}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print()

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

# Apply chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": PROMPT},
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Formatted input: {input_text[:200]}...")
input_ids = tokenizer.encode(input_text, add_special_tokens=False)
print(f"Input token count: {len(input_ids)}")
print()

# Load quantized model (seq_len=1 for decode)
print("Loading seq_len=1 model...")
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
model = Qwen3_5_0_8B_AIMETOnnx.from_pretrained(
    checkpoint=CHECKPOINT,
    sequence_length=1,
    context_length=256,
)
print("Model loaded successfully!")
print()

# Get the ONNX session
session = model.quant_sim.session

# Check model inputs
print("Model inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")
print()

# We'll do a simple greedy decode using the seq_len=1 model
# First, we need to process the prompt token-by-token through the model
# to build up the KV cache state

print("Starting generation...")
print("=" * 50)
start_time = time.time()

# For each input, prepare initial values
# The model expects: input_ids (1,1), position_ids (1,1), attention_mask,
# and past KV cache tensors

# Get input names and shapes
input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]

# Prepare inputs for first token
generated_tokens = []
all_tokens = list(input_ids)

# Process each token of the prompt through the model
print(f"Processing {len(all_tokens)} prompt tokens...")
for i, token in enumerate(all_tokens):
    feeds = {}
    for inp in session.get_inputs():
        name = inp.name
        shape = [1 if isinstance(s, str) or s is None else s for s in inp.shape]
        if name == "input_ids":
            feeds[name] = np.array([[token]], dtype=np.int64)
        elif name == "position_ids":
            feeds[name] = np.array([[i]], dtype=np.int64)
        elif "int" in inp.type.lower():
            feeds[name] = np.zeros(shape, dtype=np.int64)
        else:
            feeds[name] = np.zeros(shape, dtype=np.float32)
    
    outputs = session.run(None, feeds)
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(all_tokens)} prompt tokens")

# Get logits from last output
logits = outputs[0]  # shape: (1, 1, vocab_size) or (1, vocab_size)
if len(logits.shape) == 3:
    next_token_logits = logits[0, -1, :]
else:
    next_token_logits = logits[0, :]

next_token = int(np.argmax(next_token_logits))
generated_tokens.append(next_token)
decoded_so_far = tokenizer.decode(generated_tokens, skip_special_tokens=False)
print(f"\nFirst generated token: {next_token} -> '{tokenizer.decode([next_token])}'")

# Continue generating
pos = len(all_tokens)
for step in range(1, MAX_NEW_TOKENS):
    if next_token == tokenizer.eos_token_id:
        print(f"\n[EOS reached at step {step}]")
        break
    
    # Check for end tokens
    if next_token in [151643, 151644, 151645]:  # Qwen special tokens
        print(f"\n[Special end token {next_token} at step {step}]")
        break
    
    feeds = {}
    for inp in session.get_inputs():
        name = inp.name
        shape = [1 if isinstance(s, str) or s is None else s for s in inp.shape]
        if name == "input_ids":
            feeds[name] = np.array([[next_token]], dtype=np.int64)
        elif name == "position_ids":
            feeds[name] = np.array([[pos]], dtype=np.int64)
        elif "int" in inp.type.lower():
            feeds[name] = np.zeros(shape, dtype=np.int64)
        else:
            feeds[name] = np.zeros(shape, dtype=np.float32)
    
    outputs = session.run(None, feeds)
    logits = outputs[0]
    if len(logits.shape) == 3:
        next_token_logits = logits[0, -1, :]
    else:
        next_token_logits = logits[0, :]
    
    next_token = int(np.argmax(next_token_logits))
    generated_tokens.append(next_token)
    pos += 1
    
    if (step + 1) % 5 == 0:
        partial = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"  Step {step+1}: ...{partial[-40:]}")

elapsed = time.time() - start_time
print(f"\n{'=' * 50}")
print(f"Generation complete in {elapsed:.1f}s ({len(generated_tokens)} tokens, {len(generated_tokens)/elapsed:.1f} tok/s)")
print(f"\nFull response:")
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)
