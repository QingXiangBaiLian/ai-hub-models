"""Generate text from the quantized Qwen3.5-0.8B model with a Chinese prompt."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')

from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx

print("Loading quantized model (INT8 recalibrated, block_size=32)...")
model = Qwen3_5_0_8B_AIMETOnnx.from_pretrained(
    checkpoint='/workspace/qwen3_5_0_8b_w4a16_static',
    sequence_length=128,
    context_length=256,
)
print("Model loaded.\n")

prompt = "你是谁呀？"
print(f"Prompt: {prompt}")
print(f"Generating response...\n")

# Tokenize
input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
print(f"Input tokens: {input_ids.shape[1]}")

# Generate using the model's generate method
import torch
import numpy as np

# Use the LLM_Generator if available, otherwise manual autoregressive
try:
    from qai_hub_models.models._shared.llm.model import LLM_Generator
    
    generator = LLM_Generator(model)
    output_text = generator.generate(prompt, max_new_tokens=100)
    print(f"\nResponse:\n{output_text}")
except Exception as e:
    print(f"Generator approach failed: {e}")
    print("Trying manual generation...")
    
    # Manual autoregressive generation
    from transformers import GenerationConfig
    
    tokenizer = model.tokenizer
    tokens = tokenizer.encode(prompt)
    generated = list(tokens)
    max_new_tokens = 100
    
    print(f"Starting with {len(tokens)} tokens, generating up to {max_new_tokens} more...")
    
    for i in range(max_new_tokens):
        # The model forward expects specific inputs
        # Let's use the evaluate-style forward
        try:
            output = model.forward_token(generated)
            next_token = output.argmax(-1).item()
            generated.append(next_token)
            
            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1} tokens...")
        except Exception as e2:
            print(f"  Error at token {i}: {e2}")
            break
    
    response = tokenizer.decode(generated[len(tokens):], skip_special_tokens=True)
    print(f"\nResponse:\n{response}")
