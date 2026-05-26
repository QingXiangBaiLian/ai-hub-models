"""Diagnostic: Check if encodings are being loaded correctly into QuantSim.
Tests whether encoding names match quantizer names in the QuantSim model."""

import json
import os
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')

import onnx
from onnx.external_data_helper import load_external_data_for_model

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'
onnx_path = os.path.join(checkpoint, 'model_seqlen128_cl256.onnx')
encodings_path = os.path.join(checkpoint, 'model.encodings')

# Load ONNX model
print("Loading ONNX model...")
onnx_model = onnx.load(onnx_path, load_external_data=False)
load_external_data_for_model(onnx_model, os.path.dirname(onnx_path))

# Create QuantSim
print("Creating QuantSim...")
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision
import torch

quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(
    onnx_model, torch.device('cpu'), 
    Precision.w4a16
)

# Get all quantizer names before loading encodings
from aimet_onnx.quantsim import QuantizationSimModel
quantizer_names = set(quant_sim.qc_quantize_op_dict.keys())
print(f"\nTotal quantizers in QuantSim: {len(quantizer_names)}")

# Show first 20 quantizer names
print("\nFirst 20 quantizer names:")
for i, name in enumerate(sorted(quantizer_names)[:20]):
    print(f"  {name}")

# Load the encodings file
with open(encodings_path, 'r') as f:
    enc = json.load(f)

# Get encoding names
act_enc = enc.get('activation_encodings', [])
param_enc = enc.get('param_encodings', [])

act_names = set()
param_names = set()
for item in act_enc:
    if isinstance(item, dict) and 'name' in item:
        act_names.add(item['name'])
for item in param_enc:
    if isinstance(item, dict) and 'name' in item:
        param_names.add(item['name'])

all_encoding_names = act_names | param_names
print(f"\nTotal activation encoding names: {len(act_names)}")
print(f"Total param encoding names: {len(param_names)}")
print(f"Total encoding names: {len(all_encoding_names)}")

# Check matches
matched = quantizer_names & all_encoding_names
unmatched_quantizers = quantizer_names - all_encoding_names
unmatched_encodings = all_encoding_names - quantizer_names

print(f"\n--- Match Analysis ---")
print(f"Matched (encoding names that exist in QuantSim): {len(matched)}")
print(f"Unmatched quantizers (no encoding): {len(unmatched_quantizers)}")
print(f"Unmatched encodings (no quantizer): {len(unmatched_encodings)}")

if unmatched_encodings:
    print(f"\nFirst 20 unmatched encoding names (in file but NOT in QuantSim):")
    for name in sorted(unmatched_encodings)[:20]:
        print(f"  {name}")

if unmatched_quantizers:
    print(f"\nFirst 20 unmatched quantizer names (in QuantSim but NOT in encodings):")
    for name in sorted(unmatched_quantizers)[:20]:
        print(f"  {name}")

# Now actually load encodings and check what happens
print("\n--- Loading encodings with strict=False ---")
from aimet_onnx.quantsim import load_encodings_to_sim
load_encodings_to_sim(quant_sim, encodings_path, strict=False)

# Check quantizer states after loading
enabled_with_encoding = 0
enabled_without_encoding = 0
disabled = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if not qc_op.enabled:
        disabled += 1
    elif hasattr(qc_op, 'encodings') and qc_op.encodings is not None:
        enabled_with_encoding += 1
    else:
        enabled_without_encoding += 1

print(f"\nAfter loading encodings:")
print(f"  Enabled quantizers WITH encodings: {enabled_with_encoding}")
print(f"  Enabled quantizers WITHOUT encodings: {enabled_without_encoding}")
print(f"  Disabled quantizers: {disabled}")

# Show some quantizers without encodings
if enabled_without_encoding > 0:
    print(f"\n  First 20 enabled quantizers WITHOUT encodings:")
    count = 0
    for name, qc_op in sorted(quant_sim.qc_quantize_op_dict.items()):
        if qc_op.enabled and (not hasattr(qc_op, 'encodings') or qc_op.encodings is None):
            print(f"    {name} (bw={qc_op.bitwidth})")
            count += 1
            if count >= 20:
                break
