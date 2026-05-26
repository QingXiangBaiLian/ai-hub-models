"""Convert W4 param encodings to W8 by adjusting scales.

For symmetric quantization:
  INT4: scale = max_abs / 7
  INT8: scale = max_abs / 127
  => scale_int8 = scale_int4 * 7 / 127
"""
import json
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path) as f:
    enc = json.load(f)

converted = 0
for pe in enc["param_encodings"]:
    if pe["bw"] == 4:
        # Adjust scale for INT8
        if isinstance(pe["scale"], list):
            pe["scale"] = [s * 7.0 / 127.0 for s in pe["scale"]]
        else:
            pe["scale"] = pe["scale"] * 7.0 / 127.0
        # Offset stays 0 for symmetric
        if isinstance(pe["offset"], list):
            pe["offset"] = [0.0 for _ in pe["offset"]]
        else:
            pe["offset"] = 0.0
        pe["bw"] = 8
        converted += 1

print(f"Converted {converted} param encodings from INT4 to INT8")
sample = enc["param_encodings"][0]
print(f"Sample: name={sample['name']}, bw={sample['bw']}, is_sym={sample.get('is_sym')}")
if isinstance(sample["scale"], list):
    print(f"  scale[0:3]={sample['scale'][:3]}")

with open(output_path, "w") as f:
    json.dump(enc, f)
print(f"Saved to {output_path}")
