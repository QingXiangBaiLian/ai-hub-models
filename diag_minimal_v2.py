"""Minimal model test: verify AIMET blockwise INT4 quantization correctness."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import logging
logging.getLogger('Quant').setLevel(logging.ERROR)

# Create (64, 64) weight with known values
np.random.seed(42)
weight_data = np.random.randn(64, 64).astype(np.float32) * 0.1
input_data = np.random.randn(1, 64).astype(np.float32)

# Model: Y = X @ W.T  (via Transpose + MatMul)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 64])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 64])
W = numpy_helper.from_array(weight_data, name='W')
transpose_node = helper.make_node('Transpose', ['W'], ['W_T'], perm=[1, 0])
matmul_node = helper.make_node('MatMul', ['X', 'W_T'], ['Y'])
graph = helper.make_graph([transpose_node, matmul_node], 'test', [X], [Y], initializer=[W])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21)])
model.ir_version = 9

# FP32 baseline
sess = ort.InferenceSession(model.SerializeToString())
fp_out = sess.run(None, {'X': input_data})[0]
print(f"FP32 output: mean={fp_out.mean():.6f}, std={fp_out.std():.6f}")

# Create QuantSim
from aimet_onnx.quantsim import QuantizationSimModel, set_blockwise_quantization_for_weights

quant_sim = QuantizationSimModel(
    model=model,
    quant_scheme='min_max',
    default_param_bw=4,
    default_activation_bw=16,
)

# Set blockwise quantization
set_blockwise_quantization_for_weights(
    sim=quant_sim,
    op_types=("MatMul",),
    bitwidth=4,
    symmetric=True,
    block_size=32,
)

w_qtzr = quant_sim.qc_quantize_op_dict['W']
print(f"\nWeight quantizer after blockwise setup:")
print(f"  enabled: {w_qtzr.enabled}")
print(f"  bitwidth: {w_qtzr.bitwidth}")
print(f"  usePerChannelMode: {w_qtzr.quant_info.usePerChannelMode}")
print(f"  channelAxis: {w_qtzr.quant_info.channelAxis}")
print(f"  blockAxis: {w_qtzr.quant_info.blockAxis}")
print(f"  blockSize: {w_qtzr.quant_info.blockSize}")
if w_qtzr.tensor_quantizer_params:
    print(f"  tensor_shape: {w_qtzr.tensor_quantizer_params.tensor_shape}")

# Compute encodings 
print("\nComputing encodings...")
quant_sim.compute_encodings(lambda session, _: session.run(None, {'X': input_data}), None)

# Check computed encodings
encodings = w_qtzr.get_encodings()
print(f"Number of encodings: {len(encodings)}")
if encodings:
    print(f"First encoding: delta={encodings[0].delta:.8f}, offset={encodings[0].offset}, bw={encodings[0].bw}")
    print(f"  min={encodings[0].min:.8f}, max={encodings[0].max:.8f}")
    tf_offsets = np.array([e.offset for e in encodings])
    print(f"All offsets unique: {np.unique(tf_offsets)}")

# Run quantized inference
print("\nRunning quantized inference...")
quant_out = quant_sim.session.run(None, {'X': input_data})[0]
print(f"Quantized output: mean={quant_out.mean():.6f}, std={quant_out.std():.6f}")

# Compute output SQNR
mse = np.mean((fp_out - quant_out)**2)
signal = np.mean(fp_out**2)
sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')
print(f"Output SQNR: {sqnr:.2f} dB")

# Manual computation: quantize weight then compute output
# Weight (64, 64), channel_axis=0 or -2, block_axis=1 or -1, block_size=32
# Scales should have shape (64, 2) = 128 values
channel_axis = w_qtzr.quant_info.channelAxis
block_axis = w_qtzr.quant_info.blockAxis
block_size = w_qtzr.quant_info.blockSize
tensor_shape = w_qtzr.tensor_quantizer_params.tensor_shape
print(f"\nResolved axes: channel={channel_axis}, block={block_axis}, block_size={block_size}")
print(f"Tensor shape: {tensor_shape}")

# Manual QDQ
n_channels = tensor_shape[channel_axis % len(tensor_shape)]
n_blocks = tensor_shape[block_axis % len(tensor_shape)] // block_size
print(f"Expected: {n_channels} channels, {n_blocks} blocks/channel = {n_channels * n_blocks} total")
print(f"Actual encodings: {len(encodings)}")

# Use AIMET's computed scales
aimet_scales = np.array([e.delta for e in encodings]).reshape(n_channels, n_blocks)
aimet_offsets = np.array([e.offset for e in encodings]).reshape(n_channels, n_blocks)

# Manual QDQ with signed symmetric (offset=0 internal, using scale only)
manual_qdq = np.zeros_like(weight_data)
for i in range(n_channels):
    for j in range(n_blocks):
        s = aimet_scales[i, j]
        block = weight_data[i, j*block_size:(j+1)*block_size]
        q = np.clip(np.round(block / s), -8, 7)
        manual_qdq[i, j*block_size:(j+1)*block_size] = q * s

manual_out = input_data @ manual_qdq.T
manual_mse = np.mean((fp_out - manual_out)**2)
manual_sqnr = 10 * np.log10(signal / manual_mse) if manual_mse > 0 else float('inf')
print(f"\nManual QDQ output SQNR: {manual_sqnr:.2f} dB")

# Compare AIMET output vs manual output
diff = np.abs(quant_out - manual_out)
print(f"AIMET vs Manual max diff: {diff.max():.8f}")
print(f"AIMET vs Manual match: {np.allclose(quant_out, manual_out, atol=1e-4)}")

if not np.allclose(quant_out, manual_out, atol=1e-4):
    print("\n!!! AIMET output differs from manual computation !!!")
    print(f"AIMET out[:8]: {quant_out[0, :8]}")
    print(f"Manual out[:8]: {manual_out[0, :8]}")
    
    # Check if it matches unsigned formula instead
    manual_qdq_unsigned = np.zeros_like(weight_data)
    for i in range(n_channels):
        for j in range(n_blocks):
            s = aimet_scales[i, j]
            o = aimet_offsets[i, j]  # -8
            block = weight_data[i, j*block_size:(j+1)*block_size]
            q = np.clip(np.round(block / s - o), 0, 15)
            manual_qdq_unsigned[i, j*block_size:(j+1)*block_size] = (q + o) * s
    
    manual_unsigned_out = input_data @ manual_qdq_unsigned.T
    diff_unsigned = np.abs(quant_out - manual_unsigned_out)
    print(f"\nAIMET vs Manual (unsigned) max diff: {diff_unsigned.max():.8f}")
    print(f"AIMET vs Manual (unsigned) match: {np.allclose(quant_out, manual_unsigned_out, atol=1e-4)}")
else:
    print("\n=== AIMET quantizer produces CORRECT output ===")
    print("The bug is NOT in the quantizer itself.")
