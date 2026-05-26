"""
Verify quantizer state and test if disabling actually works.
Measure per-layer output difference between quantized and FP modes.
"""
import sys
sys.path.insert(0, "/workspace/repos/ai-hub-models/src")

import os
import json
import numpy as np
import onnx
from onnx.external_data_helper import load_external_data_for_model

ckpt = "/workspace/qwen3_5_0_8b_w4a16_static"
onnx_path = os.path.join(ckpt, "model_seqlen128_cl256.onnx")
encodings_path = os.path.join(ckpt, "model.encodings")

# Load the model
print("Loading ONNX model...")
onnx_model = onnx.load(onnx_path, load_external_data=False)
load_external_data_for_model(onnx_model, ckpt)

# Create QuantSim in the same way the model does
from aimet_onnx.quantsim import QuantizationSimModel, QuantScheme, load_encodings_to_sim
from aimet_onnx.common import quantsim as qs
from aimet_onnx import quantsim
from aimet_onnx.common.utils import AimetLogger
import logging

AimetLogger.set_level_for_all_areas(logging.WARNING)

# Get the config path
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
default_config = get_aimet_config_path("default_config_qwen35")

quantsim.op_types_to_tie_qtzrs = ["Concat"]
quantsim._tie_qtzrs = True
quantsim.op_outputs_to_ignore.append("Slice")
quantsim.op_outputs_to_ignore.append("Constant")
quantsim.op_outputs_to_ignore.append("SplitToSequence")
quantsim.op_outputs_to_ignore.append("SequenceAt")
quantsim.op_outputs_to_ignore.append("SequenceConstruct")
qs.encoding_version = "1.0.0"

if 0 not in onnx.mapping.TENSOR_TYPE_MAP:
    onnx.mapping.TENSOR_TYPE_MAP[0] = onnx.mapping.TENSOR_TYPE_MAP[onnx.TensorProto.FLOAT16]

print("Creating QuantSim...")
quant_sim = QuantizationSimModel(
    model=onnx_model,
    param_type="int4",
    activation_type="int16",
    quant_scheme=QuantScheme.min_max,
    config_file=default_config,
    providers=["CPUExecutionProvider"],
)
del onnx_model

print("Applying W4A16 configuration...")
from qai_hub_models.models._shared.llm._utils import (
    _apply_int8_kv_cache_tying_and_lm_head,
    _get_kv_io_map,
)
from qai_hub_models.models._shared.qwen3_5.model import Qwen3_5Base_AIMETOnnx

kv_io_map = _get_kv_io_map(quant_sim)
quant_sim = _apply_int8_kv_cache_tying_and_lm_head(quant_sim, kv_io_map)
Qwen3_5Base_AIMETOnnx._set_mamba_states_to_float16(quant_sim)
Qwen3_5Base_AIMETOnnx._set_int4_weights_to_per_block(quant_sim, block_size=32)

print("\n=== BEFORE loading encodings ===")
# Check activation quantizer states
act_bw_dist = {}
for name in quant_sim.activation_names:
    qc_op = quant_sim.qc_quantize_op_dict.get(name)
    if qc_op and qc_op.enabled:
        bw = qc_op.bitwidth
        act_bw_dist[bw] = act_bw_dist.get(bw, 0) + 1
print(f"Enabled activation quantizers by bitwidth: {act_bw_dist}")

param_bw_dist = {}
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name not in quant_sim.activation_names:
        if qc_op.enabled:
            bw = qc_op.bitwidth
            param_bw_dist[bw] = param_bw_dist.get(bw, 0) + 1
print(f"Enabled param quantizers by bitwidth: {param_bw_dist}")

# Load encodings
print("\nLoading encodings...")
load_encodings_to_sim(quant_sim, encodings_path, strict=False)

print("\n=== AFTER loading encodings ===")
act_bw_dist = {}
act_disabled = 0
for name in quant_sim.activation_names:
    qc_op = quant_sim.qc_quantize_op_dict.get(name)
    if qc_op:
        if qc_op.enabled:
            bw = qc_op.bitwidth
            act_bw_dist[bw] = act_bw_dist.get(bw, 0) + 1
        else:
            act_disabled += 1
print(f"Enabled activation quantizers by bitwidth: {act_bw_dist}")
print(f"Disabled activation quantizers: {act_disabled}")

param_bw_dist = {}
param_disabled = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name not in quant_sim.activation_names:
        if qc_op.enabled:
            bw = qc_op.bitwidth
            param_bw_dist[bw] = param_bw_dist.get(bw, 0) + 1
        else:
            param_disabled += 1
print(f"Enabled param quantizers by bitwidth: {param_bw_dist}")
print(f"Disabled param quantizers: {param_disabled}")

# Now apply our fixes
print("\n=== Applying fixes ===")
# Disable gate weight quantizers
gate_disabled = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names:
        continue
    if "in_proj_a" in name or "in_proj_b" in name:
        qc_op.enabled = False
        gate_disabled += 1
print(f"Disabled {gate_disabled} gate weight quantizers")

# Disable INT16 activation quantizers
int16_disabled = 0
for name in quant_sim.activation_names:
    qc_op = quant_sim.qc_quantize_op_dict.get(name)
    if qc_op and qc_op.enabled and qc_op.bitwidth == 16:
        qc_op.enabled = False
        int16_disabled += 1
print(f"Disabled {int16_disabled} INT16 activation quantizers")

print("\n=== AFTER fixes ===")
act_bw_dist = {}
act_disabled = 0
for name in quant_sim.activation_names:
    qc_op = quant_sim.qc_quantize_op_dict.get(name)
    if qc_op:
        if qc_op.enabled:
            bw = qc_op.bitwidth
            act_bw_dist[bw] = act_bw_dist.get(bw, 0) + 1
        else:
            act_disabled += 1
print(f"Enabled activation quantizers by bitwidth: {act_bw_dist}")
print(f"Disabled activation quantizers: {act_disabled}")

param_bw_dist = {}
param_disabled = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name not in quant_sim.activation_names:
        if qc_op.enabled:
            bw = qc_op.bitwidth
            param_bw_dist[bw] = param_bw_dist.get(bw, 0) + 1
        else:
            param_disabled += 1
print(f"Enabled param quantizers by bitwidth: {param_bw_dist}")
print(f"Disabled param quantizers: {param_disabled}")

# Check if rebuild is needed - run inference with and without rebuild
print("\n=== Testing if _rebuild_session matters ===")
session = quant_sim.session
inputs = session.get_inputs()
input_feed = {}
for inp in inputs:
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    if "int" in inp.type.lower():
        input_feed[inp.name] = np.zeros(shape, dtype=np.int32)
    else:
        input_feed[inp.name] = np.random.randn(*shape).astype(np.float32) * 0.01

print("Running inference BEFORE rebuild...")
out_before = session.run(None, input_feed)
logits_before = out_before[0]
print(f"  Logits shape: {logits_before.shape}, mean={logits_before.mean():.6f}, std={logits_before.std():.6f}")

print("Rebuilding session...")
quant_sim._rebuild_session()
session = quant_sim.session

print("Running inference AFTER rebuild...")
out_after = session.run(None, input_feed)
logits_after = out_after[0]
print(f"  Logits shape: {logits_after.shape}, mean={logits_after.mean():.6f}, std={logits_after.std():.6f}")

# Compare
diff = np.abs(logits_before - logits_after).mean()
print(f"\n  Mean absolute difference: {diff:.8f}")
if diff > 1e-5:
    print("  >>> SESSION REBUILD CHANGES OUTPUT - this is the bug!")
else:
    print("  >>> Session rebuild does NOT change output - quantizers are applied dynamically")
