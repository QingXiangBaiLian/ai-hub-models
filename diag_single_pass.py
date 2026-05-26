"""Run single forward pass: compare quantized vs FP32 output directly."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import numpy as np
import onnx, json, os, logging
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model
import torch
logging.getLogger('Quant').setLevel(logging.ERROR)

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'
onnx_model = onnx.load(f'{checkpoint}/model_seqlen128_cl256.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, checkpoint)

# Step 1: Run FP32 model
print("Creating FP32 ORT session...")
import onnxruntime as ort
fp_sess = ort.InferenceSession(
    f'{checkpoint}/model_seqlen128_cl256.onnx',
    providers=['CPUExecutionProvider']
)

# Create dummy inputs
input_info = {inp.name: (inp.shape, inp.type) for inp in fp_sess.get_inputs()}
print(f"Model has {len(input_info)} inputs")

# Create realistic inputs
np.random.seed(42)
dummy_inputs = {}
for name, (shape, dtype) in input_info.items():
    resolved_shape = [d if isinstance(d, int) else 1 for d in shape]
    if 'float' in dtype:
        if 'attention_mask' in name:
            # attention_mask should be zeros (no masking for first token)
            dummy_inputs[name] = np.zeros(resolved_shape, dtype=np.float32)
        elif 'embeds' in name:
            dummy_inputs[name] = np.random.randn(*resolved_shape).astype(np.float32) * 0.01
        elif 'cos' in name or 'sin' in name:
            dummy_inputs[name] = np.random.randn(*resolved_shape).astype(np.float32)
        else:
            dummy_inputs[name] = np.zeros(resolved_shape, dtype=np.float32)
    elif 'int' in dtype:
        dummy_inputs[name] = np.zeros(resolved_shape, dtype=np.int64)
    else:
        dummy_inputs[name] = np.zeros(resolved_shape, dtype=np.float32)

print(f"Running FP32 forward pass...")
fp_outputs = fp_sess.run(None, dummy_inputs)
fp_output_names = [out.name for out in fp_sess.get_outputs()]
fp_logits = fp_outputs[fp_output_names.index('logits')] if 'logits' in fp_output_names else fp_outputs[0]
print(f"FP32 logits shape: {fp_logits.shape}, range: [{fp_logits.min():.4f}, {fp_logits.max():.4f}]")
print(f"FP32 logits mean: {fp_logits.mean():.6f}, std: {fp_logits.std():.6f}")
del fp_sess

# Step 2: Create quantized model and run
print("\nCreating QuantSim...")
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision
from aimet_onnx.quantsim import load_encodings_to_sim

quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(onnx_model, torch.device('cpu'), Precision.w4a16)
load_encodings_to_sim(quant_sim, f'{checkpoint}/model.encodings', strict=False)

print("Running quantized forward pass...")
quant_outputs = quant_sim.session.run(None, dummy_inputs)
quant_output_names = [out.name for out in quant_sim.session.get_outputs()]
quant_logits = quant_outputs[quant_output_names.index('logits')] if 'logits' in quant_output_names else quant_outputs[0]
print(f"Quant logits shape: {quant_logits.shape}, range: [{quant_logits.min():.4f}, {quant_logits.max():.4f}]")
print(f"Quant logits mean: {quant_logits.mean():.6f}, std: {quant_logits.std():.6f}")

# Compare
mse = np.mean((fp_logits - quant_logits)**2)
signal = np.mean(fp_logits**2)
sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')
print(f"\nLogits SQNR: {sqnr:.2f} dB")
print(f"Logits MSE: {mse:.6f}")
print(f"Max abs diff: {np.max(np.abs(fp_logits - quant_logits)):.6f}")

# Step 3: Test with quantization removed
print("\n\nRunning with quantization REMOVED...")
from qai_hub_models.utils.quantization_aimet_onnx import remove_quantization
with remove_quantization(quant_sim):
    fp2_outputs = quant_sim.session.run(None, dummy_inputs)
    fp2_logits = fp2_outputs[quant_output_names.index('logits')] if 'logits' in quant_output_names else fp2_outputs[0]

print(f"FP32-via-remove logits range: [{fp2_logits.min():.4f}, {fp2_logits.max():.4f}]")
mse2 = np.mean((fp_logits - fp2_logits)**2)
print(f"FP32 vs FP32-via-remove MSE: {mse2:.8f} (should be ~0)")

# Step 4: Try disabling only activation quantizers
print("\n\nDisabling all activation quantizers...")
disabled_count = 0
for name, qtzr in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names and qtzr.enabled:
        qtzr.enabled = False
        disabled_count += 1
print(f"Disabled {disabled_count} activation quantizers")

# Need to check enabled weight quantizers
weight_enabled = sum(1 for name, q in quant_sim.qc_quantize_op_dict.items() 
                     if name in quant_sim.param_names and q.enabled)
print(f"Enabled weight quantizers: {weight_enabled}")

print("Running with weights-only quantization...")
wo_outputs = quant_sim.session.run(None, dummy_inputs)
wo_logits = wo_outputs[quant_output_names.index('logits')] if 'logits' in quant_output_names else wo_outputs[0]
print(f"Weights-only logits range: [{wo_logits.min():.4f}, {wo_logits.max():.4f}]")
mse_wo = np.mean((fp_logits - wo_logits)**2)
sqnr_wo = 10 * np.log10(signal / mse_wo) if mse_wo > 0 else float('inf')
print(f"Weights-only SQNR: {sqnr_wo:.2f} dB")
print(f"Weights-only max diff: {np.max(np.abs(fp_logits - wo_logits)):.6f}")
