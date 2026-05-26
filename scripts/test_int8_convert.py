"""
Test if setting qc_op.bitwidth=8 on a loaded INT4 quantizer
properly adjusts the quantization range, or if we also need
to manually adjust the scale.

Strategy: Load model through the full pipeline, check if the
bitwidth change alone produces correct output.
"""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import shutil

# Make sure original INT4 encodings are in place
shutil.copy('/workspace/qwen3_5_0_8b_w4a16_static/model_w4.encodings.bak',
            '/workspace/qwen3_5_0_8b_w4a16_static/model.encodings')
print("Restored original INT4 encodings")

# Now load the model - it will call _convert_weights_to_int8() which sets bw=8
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx

model = Qwen3_5_0_8B_AIMETOnnx.from_pretrained(
    checkpoint='/workspace/qwen3_5_0_8b_w4a16_static',
    sequence_length=128,
    context_length=256,
)

# Inspect a weight quantizer to see what happened
sim = model.quant_sim
count_4 = 0
count_8 = 0
for name in sim.param_names:
    qc_op = sim.get_qc_quantize_op(name)
    if qc_op.enabled:
        if qc_op.bitwidth == 4:
            count_4 += 1
        elif qc_op.bitwidth == 8:
            count_8 += 1

print(f"After conversion: {count_4} INT4 params, {count_8} INT8 params")

# Check encodings to understand if scale was adjusted
for name in list(sim.param_names)[:3]:
    qc_op = sim.get_qc_quantize_op(name)
    if qc_op.enabled:
        print(f"  {name}: bw={qc_op.bitwidth}")
        # Try to see encodings
        try:
            enc = qc_op.encodings
            if enc:
                print(f"    encodings: {enc}")
        except Exception as e:
            print(f"    encodings error: {e}")
        break
