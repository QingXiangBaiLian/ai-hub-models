"""Check if disabled quantizers actually take effect without rebuild."""
import sys
sys.path.insert(0, "/workspace/repos/ai-hub-models/src")

import onnx
import json
import os

# Check AIMET ONNX version and session mechanics
try:
    import aimet_onnx
    print(f"AIMET ONNX version: {aimet_onnx.__version__}")
except:
    print("Cannot determine AIMET ONNX version")

from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim

# Check if qc_op.enabled affects session directly or needs rebuild
# Look at how quantizers work
import inspect

# Check if QuantizationSimModel has a mechanism for enabled flag
if hasattr(QuantizationSimModel, '_rebuild_session'):
    print("Has _rebuild_session method")
    
# Check the qc_quantize_op_dict entry to understand mechanism
ckpt = "/workspace/qwen3_5_0_8b_w4a16_static"
onnx_path = os.path.join(ckpt, "model_seqlen128_cl256.onnx")
print(f"\nLoading ONNX model from {onnx_path}")
onnx_model = onnx.load(onnx_path, load_external_data=False)
from onnx.external_data_helper import load_external_data_for_model
load_external_data_for_model(onnx_model, ckpt)

# Check ONNX graph for quantize/dequantize nodes
node_types = {}
for node in onnx_model.graph.node:
    node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
print(f"\nONNX node types: {dict(sorted(node_types.items()))}")

# Check if there are QDQ nodes already in the ONNX
qdq_count = node_types.get("QuantizeLinear", 0) + node_types.get("DequantizeLinear", 0)
print(f"\nQDQ nodes in ONNX: {qdq_count}")

# Check if AIMET uses custom ops or standard QDQ
custom_ops = [n.op_type for n in onnx_model.graph.node if "quant" in n.op_type.lower() or "Qc" in n.op_type]
print(f"Custom quant ops: {set(custom_ops) if custom_ops else 'None'}")
