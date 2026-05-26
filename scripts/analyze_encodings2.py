"""Analyze v1.0.0 quantization encodings format."""
import json
import sys

encodings_path = sys.argv[1]
with open(encodings_path) as f:
    enc = json.load(f)

print("Version:", enc.get("version"))
print("Num activation encodings:", len(enc.get("activation_encodings", [])))
print("Num param encodings:", len(enc.get("param_encodings", [])))

# Show first few activation encodings to understand format
print("\n=== SAMPLE ACTIVATION ENCODING ===")
for a in enc.get("activation_encodings", [])[:3]:
    print(json.dumps(a, indent=2)[:500])
    print("---")

print("\n=== SAMPLE PARAM ENCODING ===")
for p in enc.get("param_encodings", [])[:3]:
    # Truncate the enc field for display
    p_copy = dict(p)
    if "enc" in p_copy and len(str(p_copy["enc"])) > 200:
        p_copy["enc"] = f"[{len(p_copy['enc'])} entries...]"
    print(json.dumps(p_copy, indent=2)[:500])
    print("---")

# Check param encoding structure more carefully
print("\n=== PARAM ENCODING DETAILS ===")
for p in enc.get("param_encodings", []):
    name = p.get("name", "")
    dtype = p.get("dtype", "?")
    enc_list = p.get("enc", [])
    if enc_list and isinstance(enc_list, list) and len(enc_list) > 0:
        first_enc = enc_list[0]
        bw = first_enc.get("bitwidth", "?")
        is_symmetric = first_enc.get("is_symmetric", "?")
        block_size = first_enc.get("block_size", None)
        scale = first_enc.get("scale", "?")
        if "in_proj" in name or "lm_head" in name:
            print(f"  {name}: dtype={dtype}, bw={bw}, sym={is_symmetric}, block={block_size}, scale={scale}")
    elif isinstance(enc_list, list) and len(enc_list) == 0:
        print(f"  {name}: dtype={dtype}, EMPTY enc list")

# Count by bitwidth in enc sub-entries
print("\n=== ACTUAL BITWIDTH DISTRIBUTION ===")
act_bw_dist = {}
for a in enc.get("activation_encodings", []):
    enc_list = a.get("enc", [])
    if enc_list and isinstance(enc_list, list) and len(enc_list) > 0:
        bw = enc_list[0].get("bitwidth", "none")
    else:
        bw = "empty"
    act_bw_dist[bw] = act_bw_dist.get(bw, 0) + 1
print("Activation:", act_bw_dist)

param_bw_dist = {}
for p in enc.get("param_encodings", []):
    enc_list = p.get("enc", [])
    if enc_list and isinstance(enc_list, list) and len(enc_list) > 0:
        bw = enc_list[0].get("bitwidth", "none")
        block = enc_list[0].get("block_size", None)
        key = f"bw{bw}"
        if block:
            key += f"_blk{block}"
    else:
        key = "empty"
    param_bw_dist[key] = param_bw_dist.get(key, 0) + 1
print("Param:", param_bw_dist)

# Check KV cache bitwidths
print("\n=== KV CACHE DETAILS ===")
for a in enc.get("activation_encodings", []):
    name = a.get("name", "")
    if "past_key" in name or "past_value" in name:
        enc_list = a.get("enc", [])
        if enc_list:
            bw = enc_list[0].get("bitwidth", "?")
            scale = enc_list[0].get("scale", "?")
            print(f"  {name}: bw={bw}, scale={scale}")
