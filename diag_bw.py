import sys, os, logging
sys.path.insert(0, '/workspace/ai-hub-models/src')
logging.disable(logging.CRITICAL)

import onnx
from onnx.external_data_helper import load_external_data_for_model
from qai_hub_models.models.qwen3_5_0_8b import Model
from qai_hub_models.models.common import Precision
import torch

print('Loading ONNX model...')
onnx_model = onnx.load('/workspace/qwen3_5_0_8b_w8a16_spinquant/model_seqlen128_cl256.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, '/workspace/qwen3_5_0_8b_w8a16_spinquant/')

print('Creating QuantSim...')
quant_sim = Model.create_quantsim(onnx_model, torch.device('cpu'), Precision.w8a16)
del onnx_model

# Check bitwidths BEFORE loading encodings
param_bw_before = {}
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name not in quant_sim.activation_names:
        bw = qc_op.bitwidth
        param_bw_before[bw] = param_bw_before.get(bw, 0) + 1
print(f'Param quantizer bitwidths BEFORE loading encodings: {param_bw_before}')

act_bw_before = {}
for name in quant_sim.activation_names:
    if name in quant_sim.qc_quantize_op_dict:
        bw = quant_sim.qc_quantize_op_dict[name].bitwidth
        act_bw_before[bw] = act_bw_before.get(bw, 0) + 1
print(f'Activation quantizer bitwidths BEFORE loading encodings: {act_bw_before}')

# Load encodings
print('Loading encodings...')
from aimet_onnx.quantsim import load_encodings_to_sim
load_encodings_to_sim(quant_sim, '/workspace/qwen3_5_0_8b_w8a16_spinquant/model.encodings', strict=False)

# Check bitwidths AFTER loading encodings
param_bw_after = {}
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name not in quant_sim.activation_names:
        bw = qc_op.bitwidth
        param_bw_after[bw] = param_bw_after.get(bw, 0) + 1
print(f'Param quantizer bitwidths AFTER loading encodings: {param_bw_after}')

act_bw_after = {}
for name in quant_sim.activation_names:
    if name in quant_sim.qc_quantize_op_dict:
        bw = quant_sim.qc_quantize_op_dict[name].bitwidth
        act_bw_after[bw] = act_bw_after.get(bw, 0) + 1
print(f'Activation quantizer bitwidths AFTER loading encodings: {act_bw_after}')

# Check if any quantizers are enabled/disabled
enabled_params = 0
disabled_params = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name not in quant_sim.activation_names:
        if qc_op.enabled:
            enabled_params += 1
        else:
            disabled_params += 1
print(f'Param quantizers: {enabled_params} enabled, {disabled_params} disabled')

print('Done.')
