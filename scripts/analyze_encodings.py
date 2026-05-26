"""Analyze quantization encodings from a checkpoint."""
import json
import sys

encodings_path = sys.argv[1]
with open(encodings_path) as f:
    enc = json.load(f)

print("Version:", enc.get("version"))
print("Num activation encodings:", len(enc.get("activation_encodings", [])))
print("Num param encodings:", len(enc.get("param_encodings", [])))

# Check what dtypes/bitwidths are used
act_bw = {}
for e in enc.get("activation_encodings", []):
    bw = e.get("bitwidth", "unknown")
    dt = e.get("dtype", "unknown")
    key = f"{dt}_{bw}"
    act_bw[key] = act_bw.get(key, 0) + 1
print("Activation bitwidth distribution:", act_bw)

param_bw = {}
for e in enc.get("param_encodings", []):
    bw = e.get("bitwidth", "unknown")
    dt = e.get("dtype", "unknown")
    is_block = "block_size" in str(e)
    key = f"{dt}_{bw}"
    if is_block:
        key += "_block"
    param_bw[key] = param_bw.get(key, 0) + 1
print("Param bitwidth distribution:", param_bw)

# Check for in_proj_a / in_proj_b entries
gate_params = [e for e in enc.get("param_encodings", []) if "in_proj_a" in e.get("name", "") or "in_proj_b" in e.get("name", "")]
print(f"\nGate param encodings (in_proj_a/b): {len(gate_params)}")
for g in gate_params[:5]:
    print(f"  {g['name']}: bw={g.get('bitwidth')}, dtype={g.get('dtype')}")

# Check INT8 KV cache activations
kv_acts = [e for e in enc.get("activation_encodings", []) if "past_key" in e.get("name", "") or "past_value" in e.get("name", "")]
print(f"\nKV cache activations: {len(kv_acts)}")
for k in kv_acts[:5]:
    print(f"  {k['name']}: bw={k.get('bitwidth')}, dtype={k.get('dtype')}")

# Check a sample of INT16 activations
int16_acts = [e for e in enc.get("activation_encodings", []) if e.get("bitwidth") == 16 and e.get("dtype") == "int"]
print(f"\nINT16 activations: {len(int16_acts)}")
if int16_acts:
    # Show scale distribution
    scales = []
    for a in int16_acts:
        enc_list = a.get("enc", [])
        if enc_list:
            s = enc_list[0].get("scale", 0)
            scales.append(s)
    if scales:
        import statistics
        print(f"  Scale: min={min(scales):.6f}, max={max(scales):.6f}, mean={statistics.mean(scales):.6f}")
        print(f"  Sample names:")
        for a in int16_acts[:10]:
            print(f"    {a['name']}: scale={a.get('enc', [{}])[0].get('scale', 'N/A')}")

# Check INT8 activations
int8_acts = [e for e in enc.get("activation_encodings", []) if e.get("bitwidth") == 8 and e.get("dtype") == "int"]
print(f"\nINT8 activations: {len(int8_acts)}")
for a in int8_acts[:5]:
    print(f"  {a['name']}: scale={a.get('enc', [{}])[0].get('scale', 'N/A')}")
