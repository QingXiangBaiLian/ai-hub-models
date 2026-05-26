"""Diagnostic: Check the actual quantizer axis and block_size settings.
Also compare the dequantized weight from QuantSim vs original."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import os
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'

# Load model and create QuantSim
print("Loading ONNX model...")
onnx_model = onnx.load(f"{checkpoint}/model_seqlen128_cl256.onnx", load_external_data=False)
load_external_data_for_model(onnx_model, os.path.dirname(f"{checkpoint}/model_seqlen128_cl256.onnx"))

# Get original weights before QuantSim modifies them
orig_weights = {}
for init in onnx_model.graph.initializer:
    if 'layers.0.mlp.gate_proj.weight' in init.name:
        orig_weights[init.name] = numpy_helper.to_array(init).copy()
        print(f"Original weight '{init.name}': shape={orig_weights[init.name].shape}")

print("\nCreating QuantSim...")
import torch
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision

quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(onnx_model, torch.device('cpu'), Precision.w4a16)

# Load encodings  
from aimet_onnx.quantsim import load_encodings_to_sim
print("\nLoading encodings...")
load_encodings_to_sim(quant_sim, f"{checkpoint}/model.encodings", strict=False)

# Check quantizer properties for gate_proj weight
print("\n--- Quantizer properties for gate_proj.weight ---")
target_name = "model.model.layers.0.mlp.gate_proj.weight"
qtzr = quant_sim.qc_quantize_op_dict.get(target_name)
if qtzr:
    print(f"  Enabled: {qtzr.enabled}")
    print(f"  Bitwidth: {qtzr.bitwidth}")
    print(f"  Symmetric: {qtzr.use_symmetric_encodings}")
    if hasattr(qtzr, 'quant_info'):
        qi = qtzr.quant_info
        print(f"  Channel axis: {qi.channelAxis if hasattr(qi, 'channelAxis') else 'N/A'}")
        print(f"  Block size: {qi.blockSize if hasattr(qi, 'blockSize') else 'N/A'}")
        print(f"  Block axis: {qi.blockAxis if hasattr(qi, 'blockAxis') else 'N/A'}")
    if hasattr(qtzr, 'tensor_quantizer_params'):
        tp = qtzr.tensor_quantizer_params
        print(f"  TQP channel_axis: {tp.channel_axis if hasattr(tp, 'channel_axis') else 'N/A'}")
        print(f"  TQP block_size: {tp.block_size if hasattr(tp, 'block_size') else 'N/A'}")
    # Try to get encodings
    try:
        enc = qtzr.get_encodings()
        if enc:
            if hasattr(enc, 'scale'):
                print(f"  Encoding scale shape: {np.array(enc.scale).shape}")
            if hasattr(enc, 'offset'):
                print(f"  Encoding offset shape: {np.array(enc.offset).shape}")
            if hasattr(enc, 'bw'):
                print(f"  Encoding bw: {enc.bw}")
            # Print some attributes
            for attr in dir(enc):
                if not attr.startswith('_'):
                    try:
                        val = getattr(enc, attr)
                        if not callable(val):
                            if isinstance(val, (int, float, bool, str)):
                                print(f"  Encoding.{attr}: {val}")
                            elif isinstance(val, (list, np.ndarray)):
                                arr = np.array(val)
                                print(f"  Encoding.{attr}: shape={arr.shape}, dtype={arr.dtype}")
                    except:
                        pass
    except Exception as e:
        print(f"  Error getting encodings: {e}")
else:
    print(f"  Quantizer not found for {target_name}!")

# Now extract the dequantized weight from the quantized model
# Run a minimal inference and check the internal weight
print("\n--- Extracting dequantized weight from QuantSim graph ---")
# After QuantSim inserts QDQ nodes, the model graph has QuantizeLinear + DequantizeLinear nodes
# The dequantized weight should be accessible

# Look for the QDQ node for gate_proj weight
model_graph = quant_sim.model.model
qdq_weight_name = target_name + "_qdq"
print(f"Looking for QDQ output: {qdq_weight_name}")

# Check DequantizeLinear nodes
dq_nodes = [n for n in model_graph.graph.node if n.op_type == "DequantizeLinear"]
print(f"Total DequantizeLinear nodes: {len(dq_nodes)}")

target_dq = None
for n in dq_nodes:
    if target_name in str(n.input) or target_name in str(n.output):
        target_dq = n
        print(f"  Found DQ node for gate_proj: inputs={list(n.input)}, outputs={list(n.output)}")
        # Check axis attribute
        for attr in n.attribute:
            print(f"    Attribute: {attr.name} = {attr.i if attr.type == 2 else attr.f if attr.type == 1 else attr.s}")
        break

if target_dq is None:
    # Try to find by checking if any DQ node takes the weight as input
    for n in dq_nodes:
        if target_name in n.input:
            target_dq = n
            print(f"  Found DQ node: inputs={list(n.input)}, outputs={list(n.output)}")
            for attr in n.attribute:
                print(f"    Attribute: {attr.name} = {attr.i if attr.type == 2 else attr.f if attr.type == 1 else attr.s}")
            break

# Also check QuantizeLinear nodes
q_nodes = [n for n in model_graph.graph.node if n.op_type == "QuantizeLinear"]
print(f"\nTotal QuantizeLinear nodes: {len(q_nodes)}")
for n in q_nodes:
    if target_name in n.input:
        print(f"  Found Q node for gate_proj: inputs={list(n.input)}, outputs={list(n.output)}")
        for attr in n.attribute:
            print(f"    Attribute: {attr.name} = {attr.i if attr.type == 2 else attr.f if attr.type == 1 else attr.s}")
        break
