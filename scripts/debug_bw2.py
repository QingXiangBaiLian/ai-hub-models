import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import shutil
import onnx
from aimet_onnx.quantsim import QuantizationSimModel

# First restore original encodings
shutil.copy('/workspace/qwen3_5_0_8b_w4a16_static/model_w4.encodings.bak',
            '/workspace/qwen3_5_0_8b_w4a16_static/model.encodings')
print("Restored original INT4 encodings")

# Load model with original INT4 encodings
model_path = '/workspace/qwen3_5_0_8b_w4a16_static/model_seqlen1_cl256.onnx'
enc_path = '/workspace/qwen3_5_0_8b_w4a16_static/model.encodings'

onnx_model = onnx.load(model_path, load_external_data=False)
sim = QuantizationSimModel(onnx_model)
from qai_hub_models.models._shared.llm.model import load_encodings_to_sim
load_encodings_to_sim(sim, enc_path, strict=False)

# Check a param quantizer
for name in list(sim.param_names)[:3]:
    qc_op = sim.get_qc_quantize_op(name)
    print(f"Param: {name}")
    print(f"  bitwidth={qc_op.bitwidth}, enabled={qc_op.enabled}")
    # Find bitwidth-related attributes
    bw_attrs = [x for x in dir(qc_op) if "bit" in x.lower() or "bw" in x.lower()]
    print(f"  bitwidth-related attrs: {bw_attrs}")
    # Try to change bitwidth
    try:
        old_bw = qc_op.bitwidth
        qc_op.bitwidth = 8
        print(f"  After setting bw=8: bitwidth={qc_op.bitwidth}")
        qc_op.bitwidth = old_bw  # restore
    except Exception as e:
        print(f"  Error setting bitwidth: {e}")
    break
