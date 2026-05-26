"""Direct test: Extract dequantized weight from QuantSim session.
Run a minimal forward pass and capture the weight after QDQ."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import os
import json
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
from onnx.external_data_helper import load_external_data_for_model
import onnxruntime as ort

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'

# Load model
print("Loading ONNX model...")
onnx_model = onnx.load(f"{checkpoint}/model_seqlen128_cl256.onnx", load_external_data=False)
load_external_data_for_model(onnx_model, os.path.dirname(f"{checkpoint}/model_seqlen128_cl256.onnx"))

# Get original weight
orig_weight = None
for init in onnx_model.graph.initializer:
    if init.name == "model.model.layers.0.mlp.gate_proj.weight":
        orig_weight = numpy_helper.to_array(init).copy()
        break
print(f"Original weight shape: {orig_weight.shape}, range: [{orig_weight.min():.4f}, {orig_weight.max():.4f}]")

# Create QuantSim and load encodings
import torch
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision
from aimet_onnx.quantsim import load_encodings_to_sim
import logging
logging.getLogger('Quant').setLevel(logging.ERROR)

print("Creating QuantSim...")
quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(onnx_model, torch.device('cpu'), Precision.w4a16)
print("Loading encodings...")
load_encodings_to_sim(quant_sim, f"{checkpoint}/model.encodings", strict=False)

# Now examine the modified ONNX graph to find the DequantizeLinear node for gate_proj
print("\nExamining QuantSim model graph...")
model_graph = quant_sim.model.model

# Find the DequantizeLinear node for gate_proj weight
target_weight = "model.model.layers.0.mlp.gate_proj.weight"
target_qdq = target_weight + "_qdq"

# Look for QDQ nodes
for node in model_graph.graph.node:
    if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
        if target_weight in node.input or target_qdq in node.output:
            print(f"\nFound {node.op_type} node:")
            print(f"  Name: {node.name}")
            print(f"  Inputs: {list(node.input)}")
            print(f"  Outputs: {list(node.output)}")
            for attr in node.attribute:
                if attr.type == 2:  # INT
                    print(f"  Attr {attr.name}: {attr.i}")
                elif attr.type == 1:  # FLOAT
                    print(f"  Attr {attr.name}: {attr.f}")
                elif attr.type == 3:  # STRING
                    print(f"  Attr {attr.name}: {attr.s}")

# Look for the scale and zero_point initializers for this weight's quantizer
scale_name = None
zp_name = None
for node in model_graph.graph.node:
    if node.op_type == "DequantizeLinear" and target_weight in node.input[0]:
        # DequantizeLinear inputs: [x, x_scale, x_zero_point]
        if len(node.input) >= 2:
            scale_name = node.input[1]
        if len(node.input) >= 3:
            zp_name = node.input[2]
        break

if scale_name:
    print(f"\nScale initializer name: {scale_name}")
    for init in model_graph.graph.initializer:
        if init.name == scale_name:
            scale_arr = numpy_helper.to_array(init)
            print(f"  Scale shape: {scale_arr.shape}")
            print(f"  Scale dtype: {scale_arr.dtype}")
            print(f"  Scale range: [{scale_arr.min():.6e}, {scale_arr.max():.6e}]")
            break

if zp_name:
    print(f"\nZero-point initializer name: {zp_name}")
    for init in model_graph.graph.initializer:
        if init.name == zp_name:
            zp_arr = numpy_helper.to_array(init)
            print(f"  ZP shape: {zp_arr.shape}")
            print(f"  ZP dtype: {zp_arr.dtype}")
            print(f"  ZP unique values: {np.unique(zp_arr)}")
            print(f"  ZP range: [{zp_arr.min()}, {zp_arr.max()}]")
            break

# Now let's manually compute what ORT's DequantizeLinear should produce
# Load the encoding for comparison
with open(f"{checkpoint}/model.encodings", 'r') as f:
    enc = json.load(f)

param_encs = {item['name']: item for item in enc['param_encodings']}
enc_info = param_encs[target_weight]
enc_scales = np.array(enc_info['scale'], dtype=np.float32)
enc_offsets = np.array(enc_info['offset'], dtype=np.float32)
print(f"\nEncoding file scales shape: {enc_scales.shape}")
print(f"Encoding file offsets unique: {np.unique(enc_offsets)}")

# Compare encoding scales with the scale in the ONNX graph
if scale_name:
    for init in model_graph.graph.initializer:
        if init.name == scale_name:
            graph_scales = numpy_helper.to_array(init).flatten()
            if graph_scales.shape == enc_scales.shape:
                diff = np.abs(graph_scales - enc_scales).max()
                print(f"\nScale difference (graph vs encoding file): max={diff:.6e}")
            else:
                print(f"\nScale shapes differ! Graph: {graph_scales.shape} vs Encoding: {enc_scales.shape}")
            break
