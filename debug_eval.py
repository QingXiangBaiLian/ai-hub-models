"""Debug script to investigate quantized model PPL issue."""
import json
import numpy as np

# Check encodings structure
with open('/workspace/qwen3_5_0_8b_w4a16_static/model.encodings') as f:
    enc = json.load(f)

print("Top keys:", list(enc.keys()))

if 'activation_encodings' in enc:
    act_keys = list(enc['activation_encodings'].keys())
    print(f"\nNum activation encodings: {len(act_keys)}")
    print("First 10:", act_keys[:10])
    # Check a sample encoding
    sample_key = act_keys[0]
    print(f"\nSample encoding for '{sample_key}':")
    print(json.dumps(enc['activation_encodings'][sample_key], indent=2)[:500])

if 'param_encodings' in enc:
    param_keys = list(enc['param_encodings'].keys())
    print(f"\nNum param encodings: {len(param_keys)}")
    print("First 10:", param_keys[:10])
    # Check a sample encoding
    sample_key = param_keys[0]
    print(f"\nSample param encoding for '{sample_key}':")
    print(json.dumps(enc['param_encodings'][sample_key], indent=2)[:500])

# Check if there are any NaN or extreme values in encodings
print("\n\n--- Checking for extreme scale values ---")
if 'activation_encodings' in enc:
    scales = []
    for k, v in enc['activation_encodings'].items():
        if isinstance(v, list):
            for entry in v:
                if 'scale' in entry:
                    scales.append(entry['scale'])
        elif isinstance(v, dict):
            if 'scale' in v:
                scales.append(v['scale'])
    if scales:
        scales_arr = np.array([s for s in scales if isinstance(s, (int, float))])
        print(f"Activation scales: min={scales_arr.min():.6e}, max={scales_arr.max():.6e}, mean={scales_arr.mean():.6e}")
        print(f"Num zero scales: {np.sum(scales_arr == 0)}")
        print(f"Num very small scales (<1e-10): {np.sum(scales_arr < 1e-10)}")
