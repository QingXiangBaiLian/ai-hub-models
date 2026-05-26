"""Test fresh INT8 weight recalibration using AIMET compute_encodings."""
import numpy as np
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')

from aimet_onnx.quantsim import QuantizationSimModel, compute_encodings
import os

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'

# Load the QuantSim directly (skip the full model pipeline)
onnx_path = os.path.join(checkpoint, 'model_seqlen1_cl256.onnx')
encodings_path = os.path.join(checkpoint, 'model.encodings')

print('Creating QuantizationSimModel...')
quant_sim = QuantizationSimModel(model=onnx_path)
print('Loading encodings...')
quant_sim.load_encodings(encodings_path, strict=False)
print(f'Loaded. Total quantizers: {len(quant_sim.qc_quantize_op_dict)}')

# Count and categorize
act_count = 0
weight_enabled = 0
weight_disabled = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names:
        act_count += 1
    else:
        if qc_op.enabled:
            weight_enabled += 1
        else:
            weight_disabled += 1

print(f'Activations: {act_count}, Weights enabled: {weight_enabled}, Weights disabled: {weight_disabled}')

# Show a few original INT4 encodings
print('\nOriginal INT4 weight encodings (first 3):')
count = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names:
        continue
    if not qc_op.enabled:
        continue
    enc = qc_op.encodings
    if enc and count < 3:
        print(f'  {name}: bw={qc_op.bitwidth}, enc[0].bw={enc[0].bw}, delta={enc[0].delta:.8f}, min={enc[0].min:.6f}, max={enc[0].max:.6f}')
        count += 1

# Step 1: Disable all activation quantizers
print('\nDisabling activation quantizers...')
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names and qc_op.enabled:
        qc_op.enabled = False

# Step 2: Disable gate weight quantizers (in_proj_a/b)
print('Disabling gate weight quantizers...')
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names:
        continue
    if 'in_proj_a' in name or 'in_proj_b' in name:
        qc_op.enabled = False

# Step 3: Set all remaining weight quantizers to bitwidth 8
print('Setting weight quantizers to bitwidth 8...')
set_count = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names:
        continue
    if not qc_op.enabled:
        continue
    qc_op.set_bitwidth(8)
    set_count += 1
print(f'Set {set_count} weight quantizers to bitwidth 8')

# Step 4: Create dummy inputs and run compute_encodings
session = quant_sim.session
print(f'\nModel inputs:')
dummy_inputs = {}
for inp in session.get_inputs():
    shape = inp.shape
    concrete_shape = []
    for s in shape:
        if isinstance(s, str) or s is None:
            concrete_shape.append(1)
        else:
            concrete_shape.append(s)
    if 'int' in inp.type.lower():
        dummy_inputs[inp.name] = np.ones(concrete_shape, dtype=np.int64)
    else:
        dummy_inputs[inp.name] = np.zeros(concrete_shape, dtype=np.float32)
    print(f'  {inp.name}: shape={concrete_shape}, type={inp.type}')

print('\nRunning compute_encodings (fresh INT8 calibration)...')
with compute_encodings(quant_sim):
    quant_sim.session.run(None, dummy_inputs)
print('Done!')

# Check new encodings
print('\nFresh INT8 weight encodings (first 3):')
count = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names:
        continue
    if not qc_op.enabled:
        continue
    enc = qc_op.encodings
    if enc and count < 3:
        print(f'  {name}: bw={qc_op.bitwidth}, enc[0].bw={enc[0].bw}, delta={enc[0].delta:.8f}, min={enc[0].min:.6f}, max={enc[0].max:.6f}')
        count += 1

print('\nRecalibration test complete!')
