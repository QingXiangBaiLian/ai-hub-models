import json, collections
with open('/workspace/qwen3_5_0_8b_w4a16_static/model.encodings') as f:
    enc = json.load(f)
print('Version:', enc.get('version'))
print('Param encodings count:', len(enc['param_encodings']))
print('Activation encodings count:', len(enc['activation_encodings']))

# Show first param encoding structure
pe = enc['param_encodings'][0]
print()
print('First param encoding keys:', list(pe.keys()))
print('First param name:', pe['name'])
print('First param enc_type:', pe.get('enc_type'))
print('First param bw:', pe.get('bw'))
print('First param block_size:', pe.get('block_size'))
print('First param is_sym:', pe.get('is_sym'))
print('First param scale length:', len(pe['scale']) if isinstance(pe['scale'], list) else 'scalar')

# Show a few different param types
enc_types = collections.Counter(p.get('enc_type') for p in enc['param_encodings'])
print('\nParam enc_types:', dict(enc_types))
bws = collections.Counter(p.get('bw') for p in enc['param_encodings'])
print('Param bitwidths:', dict(bws))
block_sizes = collections.Counter(p.get('block_size') for p in enc['param_encodings'])
print('Param block_sizes:', dict(block_sizes))

# Show conv1d encoding details
for pe in enc['param_encodings']:
    if 'conv1d' in pe['name']:
        print(f"\nConv1d: {pe['name']}")
        print(f"  enc_type: {pe.get('enc_type')}, bw: {pe.get('bw')}, block_size: {pe.get('block_size')}")
        print(f"  scale length: {len(pe['scale']) if isinstance(pe['scale'], list) else 'scalar'}")
        break

# Show lm_head encoding
for pe in enc['param_encodings']:
    if 'lm_head' in pe['name'] or 'embed' in pe['name']:
        print(f"\nLM/Embed: {pe['name']}")
        print(f"  enc_type: {pe.get('enc_type')}, bw: {pe.get('bw')}, block_size: {pe.get('block_size')}")
        print(f"  scale length: {len(pe['scale']) if isinstance(pe['scale'], list) else 'scalar'}")

# Show first 5 param encodings with their scale lengths and shapes
print("\nFirst 10 param encodings:")
for pe in enc['param_encodings'][:10]:
    scale = pe['scale']
    scale_len = len(scale) if isinstance(scale, list) else 1
    print(f"  {pe['name']}: enc_type={pe.get('enc_type')}, bw={pe.get('bw')}, bs={pe.get('block_size')}, scales={scale_len}")
