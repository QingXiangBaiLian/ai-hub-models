"""Investigate the quantization encodings file structure."""
import json
import numpy as np

# Check encodings file structure
with open('/workspace/qwen3_5_0_8b_w4a16_static/model.encodings', 'r') as f:
    enc = json.load(f)

print("Top-level keys:", list(enc.keys()))
print()

if 'activation_encodings' in enc:
    act_enc = enc['activation_encodings']
    print(f"Number of activation encodings: {len(act_enc)}")
    print(f"Type: {type(act_enc)}")
    if isinstance(act_enc, list):
        for i, item in enumerate(act_enc[:3]):
            print(f"  [{i}]: {json.dumps(item)[:300]}")
    elif isinstance(act_enc, dict):
        for i, (k, v) in enumerate(list(act_enc.items())[:3]):
            print(f"  {k}: {json.dumps(v)[:300]}")
    print("  ...")
    print()

if 'param_encodings' in enc:
    param_enc = enc['param_encodings']
    print(f"Number of param encodings: {len(param_enc)}")
    print(f"Type: {type(param_enc)}")
    if isinstance(param_enc, list):
        for i, item in enumerate(param_enc[:3]):
            print(f"  [{i}]: {json.dumps(item)[:300]}")
    elif isinstance(param_enc, dict):
        for i, (k, v) in enumerate(list(param_enc.items())[:3]):
            print(f"  {k}: {json.dumps(v)[:300]}")
    print("  ...")
    print()
    
    # Check bitwidths used
    bitwidths = set()
    dtypes = set()
    items = param_enc if isinstance(param_enc, list) else param_enc.values()
    for item in items:
        entries = item if isinstance(item, list) else [item]
        for entry in entries:
            if isinstance(entry, dict):
                if 'bitwidth' in entry:
                    bitwidths.add(entry['bitwidth'])
                if 'dtype' in entry:
                    dtypes.add(entry['dtype'])
    print(f"Param bitwidths: {bitwidths}")
    print(f"Param dtypes: {dtypes}")
    print()

# Check for other keys
for k in enc.keys():
    if k not in ['activation_encodings', 'param_encodings']:
        v = enc[k]
        if isinstance(v, dict):
            print(f"{k}: dict with {len(v)} entries")
        elif isinstance(v, list):
            print(f"{k}: list with {len(v)} entries")
        else:
            print(f"{k}: {v}")

# Check activation encodings bitwidths
if 'activation_encodings' in enc:
    act_enc = enc['activation_encodings']
    act_bitwidths = set()
    act_dtypes = set()
    items = act_enc if isinstance(act_enc, list) else act_enc.values()
    for item in items:
        entries = item if isinstance(item, list) else [item]
        for entry in entries:
            if isinstance(entry, dict):
                if 'bitwidth' in entry:
                    act_bitwidths.add(entry['bitwidth'])
                if 'dtype' in entry:
                    act_dtypes.add(entry['dtype'])
    print(f"\nActivation bitwidths: {act_bitwidths}")
    print(f"Activation dtypes: {act_dtypes}")

# Look for suspicious patterns - very large or very small scales
print("\n--- Checking for problematic encodings ---")
if 'param_encodings' in enc:
    param_enc = enc['param_encodings']
    suspicious = []
    items_iter = param_enc if isinstance(param_enc, list) else param_enc.items()
    for item in items_iter:
        if isinstance(param_enc, dict):
            k, v = item
            entries = v if isinstance(v, list) else [v]
        else:
            k = item.get('name', 'unknown')
            entries = [item]
        for entry in entries:
            if isinstance(entry, dict) and 'scale' in entry:
                scales = entry['scale'] if isinstance(entry['scale'], list) else [entry['scale']]
                max_scale = max(scales)
                min_scale = min(s for s in scales if s > 0) if any(s > 0 for s in scales) else 0
                if max_scale > 1.0 or min_scale < 1e-8:
                    suspicious.append((k, max_scale, min_scale))
    
    print(f"Param encodings with scale > 1.0 or < 1e-8: {len(suspicious)}")
    if suspicious:
        print("First 10:")
        for k, mx, mn in suspicious[:10]:
            print(f"  {k}: max_scale={mx:.6e}, min_scale={mn:.6e}")

# Check quantization_version or format info
print("\n--- Format info ---")
for k in ['version', 'quantizer_args', 'supergroups', 'config_file']:
    if k in enc:
        print(f"{k}: {enc[k]}")
