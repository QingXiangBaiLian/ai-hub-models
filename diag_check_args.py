import json
d = json.load(open('/workspace/qwen3_5_0_8b_w4a16_static/model.encodings'))
print("quantizer_args:", json.dumps(d.get('quantizer_args', {}), indent=2))
print("\nproducer:", d.get('producer'))
print("version:", d.get('version'))

# Also check: what's the average weight range vs scale?
import numpy as np
for enc in d['param_encodings'][:5]:
    scales = np.array(enc['scale'])
    print(f"\n{enc['name']}:")
    print(f"  scale stats: min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")
    # For INT4 symmetric: represented range per block = scale * 7 (max positive) to scale * (-8)
    print(f"  max representable per block: mean={scales.mean()*7:.4f}")
