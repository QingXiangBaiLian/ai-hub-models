"""Test different approaches to actually update the C++ quantizer encodings."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import shutil

# Ensure original INT4 encodings
shutil.copy('/workspace/qwen3_5_0_8b_w4a16_static/model_w4.encodings.bak',
            '/workspace/qwen3_5_0_8b_w4a16_static/model.encodings')

from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx

# Temporarily disable the auto-convert to test manually
import qai_hub_models.models._shared.qwen3_5.model as qwen_model
orig_convert = qwen_model.Qwen3_5Base_AIMETOnnx._convert_weights_to_int8
qwen_model.Qwen3_5Base_AIMETOnnx._convert_weights_to_int8 = lambda self: None

model = Qwen3_5_0_8B_AIMETOnnx.from_pretrained(
    checkpoint='/workspace/qwen3_5_0_8b_w4a16_static',
    sequence_length=128,
    context_length=256,
)

sim = model.quant_sim

# Get a weight quantizer that's still enabled (INT4)
target_name = None
target_op = None
for name, qc_op in sim.qc_quantize_op_dict.items():
    if name in sim.activation_names:
        continue
    if qc_op.enabled and qc_op.bitwidth == 4:
        target_name = name
        target_op = qc_op
        break

print(f"Target: {target_name}")
print(f"  bitwidth: {target_op.bitwidth}")
enc = target_op.encodings
print(f"  len(encodings): {len(enc)}")
print(f"  enc[0].bw={enc[0].bw}, enc[0].delta={enc[0].delta}, enc[0].min={enc[0].min}, enc[0].max={enc[0].max}")

# Test 1: Check if encodings is a property with a setter
print("\n--- Test 1: Check encodings setter ---")
try:
    # Modify and set back
    for e in enc:
        e.bw = 8
        e.delta = e.delta * 7.0 / 127.0
    target_op.encodings = enc
    target_op.bitwidth = 8
    print(f"  After setting: bitwidth={target_op.bitwidth}")
    new_enc = target_op.encodings
    print(f"  New enc[0].bw={new_enc[0].bw}, new_enc[0].delta={new_enc[0].delta}")
    print("  SUCCESS: encodings setter worked")
except Exception as e:
    print(f"  FAILED: {e}")
    
    # Test 2: Try update_quantizer_and_load_encodings
    print("\n--- Test 2: update_quantizer_and_load_encodings ---")
    try:
        # Create encoding dict format
        import inspect
        sig = inspect.signature(target_op.update_quantizer_and_load_encodings)
        print(f"  Signature: {sig}")
    except Exception as e2:
        print(f"  FAILED: {e2}")

    # Test 3: Try load_encodings
    print("\n--- Test 3: load_encodings ---")
    try:
        sig = inspect.signature(target_op.load_encodings)
        print(f"  Signature: {sig}")
    except Exception as e3:
        print(f"  FAILED: {e3}")
