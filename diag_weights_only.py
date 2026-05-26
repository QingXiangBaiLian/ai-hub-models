"""Test: disable all activation quantizers, keep only weight quantizers."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import numpy as np
import onnx, json, os, logging
from onnx.external_data_helper import load_external_data_for_model
import torch
logging.getLogger('Quant').setLevel(logging.ERROR)

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'
onnx_model = onnx.load(f'{checkpoint}/model_seqlen128_cl256.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, checkpoint)

# Step 1: Run FP32 model
print('Creating FP32 ORT session...')
import onnxruntime as ort
fp_sess = ort.InferenceSession(
    f'{checkpoint}/model_seqlen128_cl256.onnx',
    providers=['CPUExecutionProvider']
)

# Create dummy inputs
input_info = {inp.name: (inp.shape, inp.type) for inp in fp_sess.get_inputs()}
np.random.seed(42)
dummy_inputs = {}
for name, (shape, dtype) in input_info.items():
    resolved_shape = [d if isinstance(d, int) else 1 for d in shape]
    if 'float' in dtype:
        if 'attention_mask' in name:
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

print('Running FP32 forward pass...')
fp_outputs = fp_sess.run(None, dummy_inputs)
fp_output_names = [out.name for out in fp_sess.get_outputs()]
fp_logits = fp_outputs[fp_output_names.index('logits')] if 'logits' in fp_output_names else fp_outputs[0]
print(f'FP32 logits: range=[{fp_logits.min():.4f}, {fp_logits.max():.4f}], mean={fp_logits.mean():.6f}, std={fp_logits.std():.6f}')
del fp_sess

# Step 2: Create quantized model
print('\nCreating QuantSim...')
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision
from aimet_onnx.quantsim import load_encodings_to_sim

quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(onnx_model, torch.device('cpu'), Precision.w4a16)
load_encodings_to_sim(quant_sim, f'{checkpoint}/model.encodings', strict=False)

# Count quantizer types
act_enabled = sum(1 for n in quant_sim.activation_names if quant_sim.qc_quantize_op_dict.get(n) and quant_sim.qc_quantize_op_dict[n].enabled)
param_enabled = sum(1 for n in quant_sim.param_names if quant_sim.qc_quantize_op_dict.get(n) and quant_sim.qc_quantize_op_dict[n].enabled)
print(f'Enabled: {act_enabled} activation quantizers, {param_enabled} param quantizers')

# Step 3: Disable ALL activation quantizers, keep ONLY weight quantizers
print('\nDisabling all activation quantizers...')
for name in list(quant_sim.activation_names):
    qtzr = quant_sim.qc_quantize_op_dict.get(name)
    if qtzr and qtzr.enabled:
        qtzr.enabled = False

act_enabled2 = sum(1 for n in quant_sim.activation_names if quant_sim.qc_quantize_op_dict.get(n) and quant_sim.qc_quantize_op_dict[n].enabled)
param_enabled2 = sum(1 for n in quant_sim.param_names if quant_sim.qc_quantize_op_dict.get(n) and quant_sim.qc_quantize_op_dict[n].enabled)
print(f'After disable: {act_enabled2} activation, {param_enabled2} param quantizers enabled')

print('Running weights-only forward pass...')
wo_outputs = quant_sim.session.run(None, dummy_inputs)
quant_output_names = [out.name for out in quant_sim.session.get_outputs()]
wo_logits = wo_outputs[quant_output_names.index('logits')] if 'logits' in quant_output_names else wo_outputs[0]
print(f'Weights-only logits: range=[{wo_logits.min():.4f}, {wo_logits.max():.4f}], mean={wo_logits.mean():.6f}, std={wo_logits.std():.6f}')

mse_wo = np.mean((fp_logits - wo_logits)**2)
signal = np.mean(fp_logits**2)
sqnr_wo = 10 * np.log10(signal / mse_wo) if mse_wo > 0 else float('inf')
print(f'Weights-only SQNR: {sqnr_wo:.2f} dB')
print(f'Weights-only MSE: {mse_wo:.6f}')
print(f'Weights-only max abs diff: {np.max(np.abs(fp_logits - wo_logits)):.6f}')

# Step 4: Now disable ALL quantizers (everything off)
print('\nDisabling ALL quantizers...')
for name, qtzr in quant_sim.qc_quantize_op_dict.items():
    if qtzr.enabled:
        qtzr.enabled = False

print('Running fully disabled forward pass...')
fd_outputs = quant_sim.session.run(None, dummy_inputs)
fd_logits = fd_outputs[quant_output_names.index('logits')] if 'logits' in quant_output_names else fd_outputs[0]
mse_fd = np.mean((fp_logits - fd_logits)**2)
print(f'Fully disabled vs FP32 MSE: {mse_fd:.8f} (should be ~0)')
print(f'Fully disabled max diff: {np.max(np.abs(fp_logits - fd_logits)):.8f}')

print('\n=== SUMMARY ===')
print(f'Weights-only SQNR: {sqnr_wo:.2f} dB (negative=bad)')
print(f'Fully disabled MSE: {mse_fd:.8f} (should be ~0)')
