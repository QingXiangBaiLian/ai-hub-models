import json
import sys

with open("/workspace/qwen3_5_0_8b_w4a16_static/model.encodings") as f:
    enc = json.load(f)

param_encs = enc.get("param_encodings", [])
print(f"Total param encodings: {len(param_encs)}")

# Group by bitwidth
bw_groups = {}
for p in param_encs:
    name = p.get("name", "?")
    specs = p.get("enc_specs", [{}])
    if not specs:
        continue
    bw = specs[0].get("bw", "?")
    bs = specs[0].get("block_size", "?")
    dtype = specs[0].get("dtype", "?")
    key = f"bw={bw}, block_size={bs}, dtype={dtype}"
    if key not in bw_groups:
        bw_groups[key] = []
    bw_groups[key].append(name)

print("\n--- By quantization config ---")
for key, names in sorted(bw_groups.items()):
    print(f"\n{key} ({len(names)} params):")
    for n in names[:10]:
        print(f"  {n}")
    if len(names) > 10:
        print(f"  ... and {len(names)-10} more")

# Show all param names
print("\n--- All param encoding names ---")
for p in param_encs:
    name = p.get("name", "?")
    specs = p.get("enc_specs", [{}])
    bw = specs[0].get("bw", "?") if specs else "?"
    print(f"  {name}: bw={bw}")
