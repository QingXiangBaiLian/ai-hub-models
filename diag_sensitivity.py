"""Test: Disable only linear_attn gate quantizers (in_proj_a, in_proj_b)
and measure the improvement in end-to-end SQNR."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import numpy as np
import torch, onnx, logging, gc
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model
logging.getLogger('Quant').setLevel(logging.ERROR)

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'

# Load ONNX model
print("Loading ONNX model...")
onnx_model = onnx.load(f'{checkpoint}/model_seqlen128_cl256.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, checkpoint)

# Create quantsim
print("Creating QuantSim...")
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision
from aimet_onnx.quantsim import load_encodings_to_sim

quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(onnx_model, torch.device('cpu'), Precision.w4a16)
load_encodings_to_sim(quant_sim, f'{checkpoint}/model.encodings', strict=False)

# Get embedding table
init_dict = {t.name: numpy_helper.to_array(t) for t in onnx_model.graph.initializer}
embed_name = None
for name in init_dict:
    if 'embed' in name.lower():
        embed_name = name
        break
if embed_name is None:
    for name, arr in init_dict.items():
        if arr.shape == (248320, 1024):
            embed_name = name
            break
print(f"Embedding: {embed_name}, shape: {init_dict[embed_name].shape}")

# Step 1: Disable ALL activation quantizers
disabled_act = 0
for name in list(quant_sim.activation_names):
    qtzr = quant_sim.qc_quantize_op_dict.get(name)
    if qtzr and qtzr.enabled:
        qtzr.enabled = False
        disabled_act += 1
print(f"Disabled {disabled_act} activation quantizers")

# Count enabled weight quantizers before any modification
enabled_before = sum(1 for n in quant_sim.param_names 
                     if quant_sim.qc_quantize_op_dict.get(n) and quant_sim.qc_quantize_op_dict[n].enabled)
print(f"Weight quantizers enabled: {enabled_before}")

# Test 1: All weight quantizers enabled (should give 0.26 dB)
# Test 2: Disable linear_attn sensitive gates (in_proj_a, in_proj_b)
# Test 3: Disable ALL linear_attn weight quantizers

# Prepare input
input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
embed_table = torch.from_numpy(init_dict[embed_name].astype(np.float32))
input_embeds = embed_table[input_ids[0]].unsqueeze(0)

# Get model inputs
model_inputs = quant_sim.model.graph().input
input_names = [inp.name for inp in model_inputs]
print(f"Model inputs: {input_names[:5]}...")

# Prepare feed dict
feed_dict = {}
for inp in model_inputs:
    name = inp.name
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if name == input_names[0]:  # input_embeds
        feed_dict[name] = input_embeds.numpy()
    elif 'attention_mask' in name or 'mask' in name.lower():
        feed_dict[name] = np.zeros(shape, dtype=np.float32)
    elif 'position_ids' in name.lower() or 'pos' in name.lower():
        feed_dict[name] = np.arange(shape[-1], dtype=np.int64).reshape(shape)
    else:
        feed_dict[name] = np.zeros(shape, dtype=np.float32)

# Run inference and compute SQNR for different configurations
def run_and_get_logits():
    outputs = quant_sim.session.run(None, feed_dict)
    return outputs[0]  # First output is logits

# Get FP32 reference first - disable ALL weight quantizers
for name in list(quant_sim.param_names):
    qtzr = quant_sim.qc_quantize_op_dict.get(name)
    if qtzr:
        qtzr.enabled = False

print("\nRunning FP32 reference (all quantizers disabled)...")
fp32_logits = run_and_get_logits()
print(f"FP32 logits: shape={fp32_logits.shape}, range=[{fp32_logits.min():.4f}, {fp32_logits.max():.4f}]")

# Test 1: Enable ALL weight quantizers
for name in list(quant_sim.param_names):
    qtzr = quant_sim.qc_quantize_op_dict.get(name)
    if qtzr:
        qtzr.enabled = True

print("\nTest 1: ALL weight quantizers enabled...")
all_quant_logits = run_and_get_logits()
mse = np.mean((fp32_logits - all_quant_logits)**2)
signal = np.mean(fp32_logits**2)
sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')
print(f"  SQNR={sqnr:.2f} dB, MSE={mse:.6f}")

# Test 2: Disable ONLY linear_attn gate quantizers (in_proj_a, in_proj_b)
disabled_gates = 0
for name in list(quant_sim.param_names):
    if 'in_proj_a' in name or 'in_proj_b' in name:
        qtzr = quant_sim.qc_quantize_op_dict.get(name)
        if qtzr:
            qtzr.enabled = False
            disabled_gates += 1
print(f"\nTest 2: Disabled {disabled_gates} gate quantizers (in_proj_a, in_proj_b)...")
gates_disabled_logits = run_and_get_logits()
mse = np.mean((fp32_logits - gates_disabled_logits)**2)
sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')
print(f"  SQNR={sqnr:.2f} dB, MSE={mse:.6f}")

# Re-enable gates
for name in list(quant_sim.param_names):
    if 'in_proj_a' in name or 'in_proj_b' in name:
        qtzr = quant_sim.qc_quantize_op_dict.get(name)
        if qtzr:
            qtzr.enabled = True

# Test 3: Disable ALL linear_attn quantizers (keep MLP and self_attn)
disabled_linear = 0
for name in list(quant_sim.param_names):
    if 'linear_attn' in name:
        qtzr = quant_sim.qc_quantize_op_dict.get(name)
        if qtzr:
            qtzr.enabled = False
            disabled_linear += 1
print(f"\nTest 3: Disabled {disabled_linear} linear_attn quantizers...")
no_linear_logits = run_and_get_logits()
mse = np.mean((fp32_logits - no_linear_logits)**2)
sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')
print(f"  SQNR={sqnr:.2f} dB, MSE={mse:.6f}")

# Re-enable linear_attn, disable MLP and self_attn instead
for name in list(quant_sim.param_names):
    qtzr = quant_sim.qc_quantize_op_dict.get(name)
    if qtzr:
        if 'linear_attn' in name:
            qtzr.enabled = True
        elif 'mlp' in name or 'self_attn' in name:
            qtzr.enabled = False

disabled_other = sum(1 for n in quant_sim.param_names 
                     if ('mlp' in n or 'self_attn' in n) and 
                     quant_sim.qc_quantize_op_dict.get(n) and 
                     not quant_sim.qc_quantize_op_dict[n].enabled)
enabled_linear = sum(1 for n in quant_sim.param_names 
                     if 'linear_attn' in n and 
                     quant_sim.qc_quantize_op_dict.get(n) and 
                     quant_sim.qc_quantize_op_dict[n].enabled)
print(f"\nTest 4: Only linear_attn quantizers enabled ({enabled_linear}), disabled {disabled_other} others...")
only_linear_logits = run_and_get_logits()
mse = np.mean((fp32_logits - only_linear_logits)**2)
sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')
print(f"  SQNR={sqnr:.2f} dB, MSE={mse:.6f}")

# Test 5: Only MLP quantizers
for name in list(quant_sim.param_names):
    qtzr = quant_sim.qc_quantize_op_dict.get(name)
    if qtzr:
        if 'mlp' in name:
            qtzr.enabled = True
        else:
            qtzr.enabled = False

enabled_mlp = sum(1 for n in quant_sim.param_names 
                  if 'mlp' in n and 
                  quant_sim.qc_quantize_op_dict.get(n) and 
                  quant_sim.qc_quantize_op_dict[n].enabled)
print(f"\nTest 5: Only MLP quantizers enabled ({enabled_mlp})...")
only_mlp_logits = run_and_get_logits()
mse = np.mean((fp32_logits - only_mlp_logits)**2)
sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')
print(f"  SQNR={sqnr:.2f} dB, MSE={mse:.6f}")

print("\nDone!")
