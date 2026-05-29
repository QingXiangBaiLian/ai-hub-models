"""
Test FP inference on the rotated-weight ONNX model (no quantization simulation).
If PPL is bad here too, SpinQuant rotations corrupted the model.
"""
import sys, os, math
sys.path.insert(0, '/workspace/ai-hub-models/src')

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('/workspace/qwen3_5_0_8b_w8a16_spinquant')

# Load the ONNX model that has rotated weights (saved by save_calibrated_checkpoint)
# This is the model AFTER SpinQuant rotation was applied to weights
print("Loading ONNX model with rotated weights for FP inference...")

# The model_seqlen128_cl256.onnx has the QuantSim structure. 
# We need the bare model. Let's check what files are available.
import glob
files = glob.glob('/workspace/qwen3_5_0_8b_w8a16_spinquant/*.onnx')
print(f"Available ONNX files: {files}")

# Actually, save_calibrated_checkpoint saves model.onnx (renamed to model_seqlen128...) 
# under _remove_quantization_nodes() context - so model_seqlen128_cl256.onnx IS the FP model
# with rotated weights (quantization nodes removed during save).

# But wait - QuantSim export with export_model=False doesn't save model,
# then save_model_with_external_weights is called under _remove_quantization_nodes().
# So model_seqlen128_cl256.onnx should be the FP (dequantized) graph.

# Let's try to load it directly with onnxruntime (no AIMET)
model_path = '/workspace/qwen3_5_0_8b_w8a16_spinquant/model_seqlen128_cl256.onnx'
print(f"Loading {model_path} with ORT...")

try:
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print("Model loaded successfully!")
    
    # Check inputs/outputs
    print("\nModel inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("\nModel outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")
        
except Exception as e:
    print(f"Failed to load model: {e}")
    # If the QuantSim model can't be loaded directly, try loading the ONNX 
    # and stripping quantization nodes manually
    print("\nTrying alternative approach...")
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    
    model = onnx.load(model_path, load_external_data=False)
    load_external_data_for_model(model, '/workspace/qwen3_5_0_8b_w8a16_spinquant/')
    
    # Check if it has QcQuantizeOp nodes
    op_types = set(n.op_type for n in model.graph.node)
    print(f"Op types in model: {sorted(op_types)}")
    
    if 'QcQuantizeOp' in op_types:
        print("Model has QcQuantizeOp nodes - this is a QuantSim model, not bare FP")
        print("Cannot run FP inference directly on this model")
    else:
        print("Model has no QcQuantizeOp nodes - should be runnable")
