import json
from collections import Counter

with open("/workspace/qwen3_5_0_8b_w4a16_static/model.encodings") as f:
    enc = json.load(f)

configs = Counter()
names_by_type = {}
for p in enc['param_encodings']:
    key = f"bw={p['bw']}, bs={p.get('block_size','N/A')}, dtype={p['dtype']}, sym={p['is_sym']}, type={p['enc_type']}"
    configs[key] += 1
    if key not in names_by_type:
        names_by_type[key] = []
    names_by_type[key].append(p['name'])

for key, count in configs.most_common():
    print(f'{key}: {count} params')
    for n in names_by_type[key][:5]:
        print(f'  {n}')
    if count > 5:
        print(f'  ... +{count-5} more')
    print()

# Check if lm_head or embed_tokens are in param encodings
print("\n--- Special layers ---")
for p in enc['param_encodings']:
    name = p['name']
    if 'lm_head' in name or 'embed' in name or 'norm' in name:
        print(f"  {name}: bw={p['bw']}")

# Count by layer type
print("\n--- By layer component ---")
component_counts = Counter()
for p in enc['param_encodings']:
    parts = p['name'].split('.')
    # Get the component (e.g., linear_attn.in_proj_qkv)
    if 'linear_attn' in p['name']:
        comp = '.'.join([x for x in parts if x in ['linear_attn', 'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'conv1d', 'out_proj']])
    elif 'self_attn' in p['name']:
        comp = '.'.join([x for x in parts if x in ['self_attn', 'q_proj_sha', 'k_proj_sha', 'v_proj_sha', 'o_proj_conv']])
    elif 'mlp' in p['name']:
        comp = '.'.join([x for x in parts if x in ['mlp', 'gate_proj', 'up_proj', 'down_proj']])
    else:
        comp = p['name']
    component_counts[comp] += 1

for comp, count in component_counts.most_common():
    print(f"  {comp}: {count}")
