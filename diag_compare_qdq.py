"""Compare AIMET QDQ output vs numpy manual QDQ for one specific weight.
This determines if the scale ordering is correct."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import numpy as np
import json, torch, onnx, logging
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model
logging.getLogger('Quant').setLevel(logging.ERROR)

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'

# Load ONNX model
print("Loading ONNX model...")
onnx_model = onnx.load(f'{checkpoint}/model_seqlen128_cl256.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, checkpoint)
init_dict = {t.name: numpy_helper.to_array(t) for t in onnx_model.graph.initializer}

# Load encodings
with open(f'{checkpoint}/model.encodings') as f:
    enc_data = json.load(f)
param_encs = {e['name']: e for e in enc_data['param_encodings']}

# Target weight to test
target = 'model.model.layers.0.mlp.gate_proj.weight'
weight = init_dict[target].astype(np.float32)
enc = param_encs[target]
print(f"Weight: {target}, shape: {weight.shape}")
print(f"Encoding: bw={enc['bw']}, enc_type={enc['enc_type']}, block_size={enc['block_size']}, is_sym={enc['is_sym']}")
print(f"Scale len: {len(enc['scale'])}, Offset len: {len(enc['offset'])}")

# NUMPY QDQ (my implementation)
scale = np.array(enc['scale'], dtype=np.float32)
offset = np.array(enc['offset'], dtype=np.float32)
block_size = enc['block_size']

# Weight (3584, 1024), channel_axis=0, block_axis=1, block_size=32
# Reshape to (3584, 32, 32)
w_blocks = weight.reshape(weight.shape[0], weight.shape[1] // block_size, block_size)
scale_3d = scale.reshape(weight.shape[0], weight.shape[1] // block_size, 1)
offset_3d = offset.reshape(weight.shape[0], weight.shape[1] // block_size, 1)

# QDQ formula (unsigned with offset=-8 for signed symmetric)
quantized = np.clip(np.round(w_blocks / scale_3d - offset_3d), 0, 2**4 - 1)
numpy_qdq = ((quantized + offset_3d) * scale_3d).reshape(weight.shape)

numpy_mse = np.mean((weight - numpy_qdq)**2)
numpy_signal = np.mean(weight**2)
numpy_sqnr = 10 * np.log10(numpy_signal / numpy_mse)
print(f"\nNumpy QDQ: SQNR={numpy_sqnr:.2f} dB, max_diff={np.max(np.abs(weight - numpy_qdq)):.6f}")

# AIMET QDQ
print("\nCreating QuantSim...")
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision
from aimet_onnx.quantsim import load_encodings_to_sim

quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(onnx_model, torch.device('cpu'), Precision.w4a16)
load_encodings_to_sim(quant_sim, f'{checkpoint}/model.encodings', strict=False)

# Get AIMET's QDQ output for this weight
qtzr = quant_sim.qc_quantize_op_dict[target]
weight_tensor = torch.from_numpy(weight.copy())
aimet_result = qtzr.quant_info.tensorQuantizerRef.quantizeDequantize(weight_tensor)
aimet_qdq = aimet_result if isinstance(aimet_result, np.ndarray) else aimet_result.numpy()

aimet_mse = np.mean((weight - aimet_qdq)**2)
aimet_sqnr = 10 * np.log10(numpy_signal / aimet_mse)
print(f"AIMET QDQ: SQNR={aimet_sqnr:.2f} dB, max_diff={np.max(np.abs(weight - aimet_qdq)):.6f}")

# Compare numpy vs AIMET
diff = np.abs(numpy_qdq - aimet_qdq)
print(f"\nNumpy vs AIMET diff: max={diff.max():.8f}, mean={diff.mean():.8f}, nonzero={np.count_nonzero(diff)}/{diff.size}")

if diff.max() < 1e-6:
    print("MATCH: Numpy and AIMET produce same QDQ result")
else:
    print("MISMATCH: Numpy and AIMET produce DIFFERENT results!")
    # Find where they differ
    mismatch_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"  Worst mismatch at index {mismatch_idx}:")
    print(f"    Weight value: {weight[mismatch_idx]:.8f}")
    print(f"    Numpy QDQ: {numpy_qdq[mismatch_idx]:.8f}")
    print(f"    AIMET QDQ: {aimet_qdq[mismatch_idx]:.8f}")
    
    # Check first few blocks
    print("\n  First 5 blocks comparison (channel 0):")
    for b in range(5):
        s = b * block_size
        e = s + block_size
        np_block = numpy_qdq[0, s:e]
        ai_block = aimet_qdq[0, s:e]
        w_block = weight[0, s:e]
        print(f"    Block {b}: weight[0:3]={w_block[:3]}, numpy_qdq[0:3]={np_block[:3]}, aimet_qdq[0:3]={ai_block[:3]}")
    
    # Check if AIMET matches a transposed or permuted version
    # Try: what if AIMET uses block_axis=0, channel_axis=1?
    w_blocks_alt = weight.reshape(weight.shape[0] // block_size, block_size, weight.shape[1])
    scale_alt = scale.reshape(weight.shape[0] // block_size, 1, weight.shape[1])
    offset_alt = offset.reshape(weight.shape[0] // block_size, 1, weight.shape[1])
    quantized_alt = np.clip(np.round(w_blocks_alt / scale_alt - offset_alt), 0, 15)
    numpy_qdq_alt = ((quantized_alt + offset_alt) * scale_alt).reshape(weight.shape)
    diff_alt = np.abs(numpy_qdq_alt - aimet_qdq)
    print(f"\n  Alt (axes swapped) vs AIMET: max_diff={diff_alt.max():.8f}, mean={diff_alt.mean():.8f}")

# Also compare AIMET's TfEncoding ordering vs file ordering
print("\nEncoding ordering check:")
encs = qtzr.quant_info.tensorQuantizerRef.getEncodings()
print(f"  Num AIMET TfEncodings: {len(encs)}")
print(f"  File scale[0:5]: {scale[:5]}")
print(f"  AIMET enc[0:5] delta: {[encs[i].delta for i in range(5)]}")
print(f"  File scale[-5:]: {scale[-5:]}")
print(f"  AIMET enc[-5:] delta: {[encs[i].delta for i in range(len(encs)-5, len(encs))]}")

# Check if they match in order
file_scales = scale
aimet_scales = np.array([e.delta for e in encs])
if np.allclose(file_scales, aimet_scales, rtol=1e-5):
    print("  Scale ordering: MATCH (same order)")
else:
    # Check if reversed
    if np.allclose(file_scales, aimet_scales[::-1], rtol=1e-5):
        print("  Scale ordering: REVERSED!")
    else:
        # Check if transposed
        try:
            file_2d = file_scales.reshape(weight.shape[0], -1)
            aimet_2d = aimet_scales.reshape(weight.shape[0], -1)
            if np.allclose(file_2d.T.flatten(), aimet_scales, rtol=1e-5):
                print("  Scale ordering: TRANSPOSED!")
            else:
                corr = np.corrcoef(file_scales[:1000], aimet_scales[:1000])[0,1]
                print(f"  Scale ordering: MISMATCH (corr={corr:.4f})")
        except:
            print("  Scale ordering: MISMATCH (reshape failed)")
