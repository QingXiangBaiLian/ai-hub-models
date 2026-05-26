import json, numpy as np
d = json.load(open('/workspace/qwen3_5_0_8b_w4a16_static/model.encodings'))
# Check first 3 param encodings
for enc in d['param_encodings'][:3]:
    offsets = np.array(enc['offset'])
    print(f"Name: {enc['name']}")
    print(f"  offset unique: {np.unique(offsets)}, len: {len(offsets)}, block_size: {enc.get('block_size',0)}")
# Check a conv1d encoding
for enc in d['param_encodings']:
    if 'conv1d' in enc['name']:
        offsets = np.array(enc['offset'])
        print(f"Name: {enc['name']}")
        print(f"  offset unique: {np.unique(offsets)}, len: {len(offsets)}, enc_type: {enc.get('enc_type')}")
        break
