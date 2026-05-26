"""Check how to access and modify encodings (scale) on qc_quantize_op."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import shutil

# Ensure original INT4 encodings
shutil.copy('/workspace/qwen3_5_0_8b_w4a16_static/model_w4.encodings.bak',
            '/workspace/qwen3_5_0_8b_w4a16_static/model.encodings')

from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx

model = Qwen3_5_0_8B_AIMETOnnx.from_pretrained(
    checkpoint='/workspace/qwen3_5_0_8b_w4a16_static',
    sequence_length=128,
    context_length=256,
)

sim = model.quant_sim

# Inspect a weight quantizer's encoding
for name, qc_op in sim.qc_quantize_op_dict.items():
    if name in sim.activation_names:
        continue
    if not qc_op.enabled:
        continue
    print(f"Name: {name}")
    print(f"  bitwidth: {qc_op.bitwidth}")
    
    # Check encoding-related attributes
    enc_attrs = [x for x in dir(qc_op) if 'encod' in x.lower() or 'scale' in x.lower() or 'offset' in x.lower()]
    print(f"  encoding/scale/offset attrs: {enc_attrs}")
    
    # Try to access encodings
    try:
        enc = qc_op.encodings
        print(f"  encodings type: {type(enc)}")
        if enc is not None:
            if hasattr(enc, '__len__'):
                print(f"  encodings len: {len(enc)}")
                if len(enc) > 0:
                    e0 = enc[0]
                    print(f"  enc[0] type: {type(e0)}")
                    print(f"  enc[0] attrs: {[x for x in dir(e0) if not x.startswith('_')]}")
                    if hasattr(e0, 'scale'):
                        print(f"  enc[0].scale: {e0.scale}")
                    if hasattr(e0, 'bw'):
                        print(f"  enc[0].bw: {e0.bw}")
                    if hasattr(e0, 'bitwidth'):
                        print(f"  enc[0].bitwidth: {e0.bitwidth}")
            else:
                print(f"  encodings: {enc}")
                print(f"  encodings attrs: {[x for x in dir(enc) if not x.startswith('_')]}")
    except Exception as e:
        print(f"  encodings error: {e}")
    break
