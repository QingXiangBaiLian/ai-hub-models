"""Direct test of AIMET quantizer output vs manual computation."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import os, onnx, logging, json
import numpy as np
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model
import torch
logging.getLogger('Quant').setLevel(logging.ERROR)

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'
onnx_model = onnx.load(f'{checkpoint}/model_seqlen128_cl256.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, checkpoint)

# Get original weight
weight_name = 'model.model.layers.0.mlp.gate_proj.weight'
orig_weight = None
for init in onnx_model.graph.initializer:
    if init.name == weight_name:
        orig_weight = numpy_helper.to_array(init)
        break
print(f"Original weight shape: {orig_weight.shape}, dtype: {orig_weight.dtype}")
print(f"Weight range: [{orig_weight.min():.6f}, {orig_weight.max():.6f}]")

from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision
from aimet_onnx.quantsim import load_encodings_to_sim

quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(onnx_model, torch.device('cpu'), Precision.w4a16)
load_encodings_to_sim(quant_sim, f'{checkpoint}/model.encodings', strict=False)

# Get the quantizer for this weight
qtzr = quant_sim.qc_quantize_op_dict[weight_name]
print(f"\nQuantizer enabled: {qtzr.enabled}")
print(f"Quantizer bitwidth: {qtzr.bitwidth}")
print(f"Quantizer op_mode: {qtzr.op_mode}")
print(f"Quantizer data_type: {qtzr.data_type}")

# Get TfEncoding
encodings = qtzr.get_encodings()
print(f"\nNumber of TfEncoding objects: {len(encodings)}")
print(f"First encoding: delta={encodings[0].delta:.8f}, offset={encodings[0].offset}, bw={encodings[0].bw}, min={encodings[0].min:.8f}, max={encodings[0].max:.8f}")
print(f"Second encoding: delta={encodings[1].delta:.8f}, offset={encodings[1].offset}")

# Check quant_info properties 
qi = qtzr.quant_info
print(f"\nquant_info.usePerChannelMode: {qi.usePerChannelMode}")
print(f"quant_info.channelAxis: {qi.channelAxis}")
print(f"quant_info.blockSize: {qi.blockSize}")
print(f"quant_info.blockAxis: {qi.blockAxis}")
print(f"quant_info.opMode: {qi.opMode}")

# Get quantized weight from QuantSim model
# The weight in the QuantSim model should be the same as original (quantization happens in the custom op)
quant_model = quant_sim.model.model
quant_weight = None
for init in quant_model.graph.initializer:
    if init.name == weight_name:
        quant_weight = numpy_helper.to_array(init)
        break
print(f"\nWeight in QuantSim model (should be FP32 original): shape={quant_weight.shape}")
print(f"QuantSim weight == original weight: {np.allclose(quant_weight, orig_weight)}")

# Now manually compute what quantization SHOULD produce
# Load scales from encoding file for comparison
with open(f'{checkpoint}/model.encodings') as f:
    enc_file = json.load(f)
param_encs = {e['name']: e for e in enc_file['param_encodings']}
gate_enc = param_encs[weight_name]
file_scales = np.array(gate_enc['scale'], dtype=np.float64)
file_offsets = np.array(gate_enc['offset'], dtype=np.float64)
block_size = gate_enc['block_size']
print(f"\nEncoding file: block_size={block_size}, num_scales={len(file_scales)}")
print(f"File offsets unique: {np.unique(file_offsets)}")

# Compare file scales with TfEncoding deltas
tf_deltas = np.array([e.delta for e in encodings])
tf_offsets = np.array([e.offset for e in encodings])
print(f"\nTfEncoding deltas match file scales: {np.allclose(tf_deltas, file_scales)}")
print(f"TfEncoding offsets unique: {np.unique(tf_offsets)}")

# Now use the quantizer's quantize_dequantize to get actual output
# We need to run the model through ORT to see the quantized output
# Let's use AIMET's tensor_quantizer directly
print("\n--- Running quantize-dequantize through AIMET ---")
# Find the quantized output name
qdq_name = weight_name + "_qdq"
print(f"Looking for output name: {qdq_name}")

# Run a single forward pass and capture the quantized weight
# Let's try using the quantizer's internal method
from aimet_onnx.qc_quantize_op import OpMode
print(f"OpMode: {qtzr.op_mode}")

# Instead of running inference, let's use the tensor_quantizer directly
tq = qtzr._tensor_quantizer
print(f"TensorQuantizer type: {type(tq)}")

# Try to quantize a small slice manually using AIMET
# Take first row, first block of 32 elements
test_weight = orig_weight[0, :32].copy().astype(np.float32)
print(f"\nTest weight (first row, first block): shape={test_weight.shape}")
print(f"Test weight values: {test_weight[:8]}")
print(f"Test weight range: [{test_weight.min():.6f}, {test_weight.max():.6f}]")

# Encoding for this block (first block of first row)
block_scale = encodings[0].delta
block_offset = encodings[0].offset  
print(f"\nBlock scale: {block_scale:.8f}, Block offset: {block_offset}")

# Manual quantize-dequantize (correct formula)
# Formula: x_q = round(x/scale - offset).clamp(qmin, qmax)
# For unsigned (offset=-8): x_q = round(x/scale + 8).clamp(0, 15)
# Dequant: x_dq = (x_q + offset) * scale = (x_q - 8) * scale
q_unsigned = np.clip(np.round(test_weight / block_scale - block_offset), 0, 15)
dq_correct = (q_unsigned + block_offset) * block_scale
err_correct = np.mean((test_weight - dq_correct)**2)
snr_correct = 10 * np.log10(np.mean(test_weight**2) / err_correct) if err_correct > 0 else float('inf')
print(f"\nManual QDQ (unsigned with offset={block_offset}):")
print(f"  Quantized values (first 8): {q_unsigned[:8]}")
print(f"  Dequantized values (first 8): {dq_correct[:8]}")
print(f"  MSE: {err_correct:.8f}, SQNR: {snr_correct:.2f} dB")

# Alternative: signed with offset=0 (which is what the encoding SHOULD be internally)
q_signed = np.clip(np.round(test_weight / block_scale), -8, 7)
dq_signed = q_signed * block_scale
err_signed = np.mean((test_weight - dq_signed)**2)
snr_signed = 10 * np.log10(np.mean(test_weight**2) / err_signed) if err_signed > 0 else float('inf')
print(f"\nManual QDQ (signed with offset=0):")
print(f"  Quantized values (first 8): {q_signed[:8]}")
print(f"  Dequantized values (first 8): {dq_signed[:8]}")
print(f"  MSE: {err_signed:.8f}, SQNR: {snr_signed:.2f} dB")

# Now run actual AIMET quantization
# Create a small session to run just the quantizer
print("\n--- Testing actual AIMET custom op output ---")
# We'll run the full model once and extract the quantized weight
import onnxruntime as ort

# Get model input info
sess = quant_sim.session
input_names = [i.name for i in sess.get_inputs()]
print(f"Model inputs: {input_names[:5]}")

# Run model with dummy inputs
input_shapes = {i.name: i.shape for i in sess.get_inputs()}
print(f"Input shapes: {list(input_shapes.items())[:3]}")
