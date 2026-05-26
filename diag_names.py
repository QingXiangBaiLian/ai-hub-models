"""Quick check: ONNX initializer names vs encoding names."""
import onnx
import json

m = onnx.load("/workspace/qwen3_5_0_8b_w4a16_static/model_seqlen128_cl256.onnx", load_external_data=False)
init_names = sorted([w.name for w in m.graph.initializer])
print(f"Total ONNX initializers: {len(init_names)}")
print("\nFirst 30 initializer names:")
for n in init_names[:30]:
    print(f"  {n}")

# Load encodings
with open("/workspace/qwen3_5_0_8b_w4a16_static/model.encodings", "r") as f:
    enc = json.load(f)

param_enc_names = sorted([item["name"] for item in enc["param_encodings"]])
print(f"\nTotal param encoding names: {len(param_enc_names)}")
print("\nFirst 30 param encoding names:")
for n in param_enc_names[:30]:
    print(f"  {n}")

# Check overlap
init_set = set(init_names)
enc_set = set(param_enc_names)
matched = init_set & enc_set
print(f"\nMatched param names: {len(matched)} / {len(enc_set)}")
unmatched_enc = enc_set - init_set
if unmatched_enc:
    print(f"Encoding names NOT in ONNX model ({len(unmatched_enc)}):")
    for n in sorted(unmatched_enc)[:20]:
        print(f"  {n}")
