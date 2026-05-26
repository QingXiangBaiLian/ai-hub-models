import json
with open('/workspace/qwen3_5_0_8b_w4a16_static/model.encodings') as f:
    enc = json.load(f)
targets = ['lm_head', 'in_proj_a', 'in_proj_b', 'in_proj_c', 'gate_proj', 'short_conv', 'o_proj', 'v_proj']
for t in targets:
    matches = [e for e in enc['param_encodings'] if t in e['name']]
    if matches:
        e = matches[0]
        print(f"  {e['name']}: bw={e['bw']}, type={e['enc_type']}, bs={e.get('block_size','NA')}, scales={len(e['scale'])}")
        print(f"    count={len(matches)}")
    else:
        print(f"  {t}: NOT FOUND")

# Also check: how many 8-bit vs 4-bit encodings
bw_counts = {}
for e in enc['param_encodings']:
    bw = e['bw']
    bw_counts[bw] = bw_counts.get(bw, 0) + 1
print(f"\nBitwidth distribution: {bw_counts}")

# Check activation encodings
if 'activation_encodings' in enc:
    act_bw = {}
    for e in enc['activation_encodings']:
        bw = e['bw']
        act_bw[bw] = act_bw.get(bw, 0) + 1
    print(f"Activation bitwidth distribution: {act_bw}")
