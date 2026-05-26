"""Direct test: create a minimal quantizer and test its output."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import logging
logging.getLogger('Quant').setLevel(logging.ERROR)

# Create a minimal model: single MatMul with a constant weight
weight_data = np.random.randn(64, 32).astype(np.float32) * 0.1
input_data = np.random.randn(1, 32).astype(np.float32)

# Create ONNX model
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 32])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 64])
W = numpy_helper.from_array(weight_data, name='W')

# Transpose W then MatMul: Y = X @ W.T
transpose_node = helper.make_node('Transpose', ['W'], ['W_T'], perm=[1, 0])
matmul_node = helper.make_node('MatMul', ['X', 'W_T'], ['Y'])

graph = helper.make_graph(
    [transpose_node, matmul_node],
    'test_model',
    [X],
    [Y],
    initializer=[W]
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21)])
model.ir_version = 9

print("Model created. Running FP32 baseline...")
import onnxruntime as ort
sess = ort.InferenceSession(model.SerializeToString())
fp_out = sess.run(None, {'X': input_data})[0]
print(f"FP32 output: mean={fp_out.mean():.6f}, std={fp_out.std():.6f}")

# Now create QuantSim with INT4 blockwise
print("\nCreating QuantSim...")
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_common import libpymo

quant_sim = QuantizationSimModel(
    model=model,
    quant_scheme='min_max',
    default_param_bw=4,
    default_activation_bw=16,
)

print(f"Quantizers: {list(quant_sim.qc_quantize_op_dict.keys())}")

# Get the weight quantizer
w_qtzr = quant_sim.qc_quantize_op_dict.get('W')
print(f"\nWeight quantizer: {w_qtzr}")
print(f"  enabled: {w_qtzr.enabled}")
print(f"  bitwidth: {w_qtzr.bitwidth}")
print(f"  op_mode: {w_qtzr.op_mode}")

# Check tensor_quantizer_params
if w_qtzr.tensor_quantizer_params:
    ttp = w_qtzr.tensor_quantizer_params
    print(f"  tensor_shape: {ttp.tensor_shape}")
    print(f"  channel_axis: {ttp.channel_axis}")
    print(f"  block_axis: {ttp.block_axis}")

# Enable per-channel blockwise quantization 
from aimet_onnx.quantsim import set_blockwise_quantization_for_weights
set_blockwise_quantization_for_weights(quant_sim, block_size=32)

print(f"\nAfter setting blockwise:")
print(f"  quant_info.usePerChannelMode: {w_qtzr.quant_info.usePerChannelMode}")
print(f"  quant_info.channelAxis: {w_qtzr.quant_info.channelAxis}")
print(f"  quant_info.blockAxis: {w_qtzr.quant_info.blockAxis}")
print(f"  quant_info.blockSize: {w_qtzr.quant_info.blockSize}")

# Set symmetric
w_qtzr.use_symmetric_encodings = True

# Manually create encodings that match our pattern (calibrated from weight data)
# For each block of 32 along block_axis, scale = max_abs / 7
weight_reshaped = weight_data.reshape(64, 1, 32)  # (channels, blocks, block_size)
# Actually for (64, 32) with channel_axis=0, block_axis=1, block_size=32:
# only 1 block per channel since 32/32=1
# Let me use a weight of shape (64, 64) with block_size=32 instead

# Recreate with larger weight
print("\n\n=== Recreating with (64, 64) weight for 2 blocks per channel ===")
weight_data = np.random.randn(64, 64).astype(np.float32) * 0.1
input_data = np.random.randn(1, 64).astype(np.float32)

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 64])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 64])
W = numpy_helper.from_array(weight_data, name='W')
transpose_node = helper.make_node('Transpose', ['W'], ['W_T'], perm=[1, 0])
matmul_node = helper.make_node('MatMul', ['X', 'W_T'], ['Y'])
graph = helper.make_graph([transpose_node, matmul_node], 'test_model', [X], [Y], initializer=[W])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21)])
model.ir_version = 9

# FP32 baseline
sess = ort.InferenceSession(model.SerializeToString())
fp_out = sess.run(None, {'X': input_data})[0]
print(f"FP32 output: mean={fp_out.mean():.6f}, std={fp_out.std():.6f}")

# Create QuantSim
quant_sim = QuantizationSimModel(
    model=model,
    quant_scheme='min_max',
    default_param_bw=4,
    default_activation_bw=16,
)

# Set blockwise
set_blockwise_quantization_for_weights(quant_sim, block_size=32)

w_qtzr = quant_sim.qc_quantize_op_dict['W']
w_qtzr.use_symmetric_encodings = True

# Compute encodings by running calibration
print("Computing encodings...")
quant_sim.compute_encodings(lambda session, _: session.run(None, {'X': input_data}), None)

# Get the computed encodings
encodings = w_qtzr.get_encodings()
print(f"\nNumber of encodings: {len(encodings)}")
if encodings:
    print(f"First encoding: delta={encodings[0].delta:.8f}, offset={encodings[0].offset}, bw={encodings[0].bw}")
    print(f"  min={encodings[0].min:.8f}, max={encodings[0].max:.8f}")
    if len(encodings) > 1:
        print(f"Second encoding: delta={encodings[1].delta:.8f}, offset={encodings[1].offset}")

# Run quantized inference
print("\nRunning quantized inference...")
quant_out = quant_sim.session.run(None, {'X': input_data})[0]
print(f"Quantized output: mean={quant_out.mean():.6f}, std={quant_out.std():.6f}")

# Compute error
mse = np.mean((fp_out - quant_out)**2)
signal = np.mean(fp_out**2)
sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')
print(f"\nOutput SQNR: {sqnr:.2f} dB")
print(f"Output MSE: {mse:.8f}")
print(f"Output max error: {np.max(np.abs(fp_out - quant_out)):.6f}")

# Manual computation for comparison
# Weight shape (64, 64), channel_axis=0, block_axis=1, block_size=32
# scales shape should be (64, 2)
manual_scales = np.zeros((64, 2), dtype=np.float32)
for i in range(64):
    for j in range(2):
        block = weight_data[i, j*32:(j+1)*32]
        manual_scales[i, j] = np.max(np.abs(block)) / 7.0

# Manual quantize-dequantize
manual_qdq = np.zeros_like(weight_data)
for i in range(64):
    for j in range(2):
        block = weight_data[i, j*32:(j+1)*32]
        s = manual_scales[i, j]
        q = np.clip(np.round(block / s), -8, 7)
        manual_qdq[i, j*32:(j+1)*32] = q * s

manual_out = input_data @ manual_qdq.T
manual_mse = np.mean((fp_out - manual_out)**2)
manual_sqnr = 10 * np.log10(signal / manual_mse) if manual_mse > 0 else float('inf')
print(f"\nManual QDQ output SQNR: {manual_sqnr:.2f} dB")
print(f"Manual output mean={manual_out.mean():.6f}, std={manual_out.std():.6f}")

# Compare AIMET scales with manual scales
aimet_deltas = np.array([e.delta for e in encodings])
aimet_offsets = np.array([e.offset for e in encodings])
print(f"\nAIMET scales: {aimet_deltas[:4]}")
print(f"Manual scales (flat): {manual_scales.flatten()[:4]}")
print(f"AIMET offsets unique: {np.unique(aimet_offsets)}")
print(f"Scales match: {np.allclose(aimet_deltas, manual_scales.flatten(), rtol=1e-5)}")
