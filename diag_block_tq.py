"""Test BlockTensorQuantizer output directly on a weight tensor."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx.external_data_helper import load_external_data_for_model
import logging
logging.getLogger('Quant').setLevel(logging.ERROR)

# Load the real model and quantizer
checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'
onnx_model = onnx.load(f'{checkpoint}/model_seqlen128_cl256.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, checkpoint)

# Get original weight
weight_name = 'model.model.layers.0.mlp.gate_proj.weight'
orig_weight = None
for init in onnx_model.graph.initializer:
    if init.name == weight_name:
        orig_weight = numpy_helper.to_array(init)
        break
print(f"Original weight shape: {orig_weight.shape}")

import torch
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision
from aimet_onnx.quantsim import load_encodings_to_sim

quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(onnx_model, torch.device('cpu'), Precision.w4a16)
load_encodings_to_sim(quant_sim, f'{checkpoint}/model.encodings', strict=False)

# Get quantizer
qtzr = quant_sim.qc_quantize_op_dict[weight_name]
tq = qtzr._tensor_quantizer  # BlockTensorQuantizer

# Check what methods BlockTensorQuantizer has
print(f"\nBlockTensorQuantizer type: {type(tq)}")
print(f"Methods: {[m for m in dir(tq) if not m.startswith('__')]}")

# Try to use quantizeDequantize method
print(f"\nTrying to call quantizeDequantize on the weight tensor...")
test_weight = orig_weight.copy()

try:
    # Try quantizeDequantize 
    result = tq.quantizeDequantize(test_weight, qtzr.quant_info.opMode)
    print(f"quantizeDequantize result shape: {result.shape}")
    print(f"First 8 values: {result[0, :8]}")
    
    # Compare with manual
    from json import load as json_load
    import json
    with open(f'{checkpoint}/model.encodings') as f:
        enc_file = json.load(f)
    param_encs = {e['name']: e for e in enc_file['param_encodings']}
    gate_enc = param_encs[weight_name]
    scales = np.array(gate_enc['scale']).reshape(3584, 32)
    
    # Manual QDQ for first row, first block
    block = orig_weight[0, :32]
    s = scales[0, 0]
    manual_qdq = np.clip(np.round(block / s), -8, 7) * s
    print(f"\nManual QDQ first 8 values: {manual_qdq[:8]}")
    print(f"AIMET QDQ first 8 values: {result[0, :8]}")
    print(f"Match: {np.allclose(manual_qdq[:8], result[0, :8], atol=1e-6)}")
    
    # Overall error
    mse = np.mean((orig_weight - result)**2)
    signal = np.mean(orig_weight**2)
    sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')
    print(f"\nFull weight SQNR from AIMET QDQ: {sqnr:.2f} dB")
    
    # Check a few blocks across the weight
    for row in [0, 100, 1000, 3000]:
        for blk in [0, 15, 31]:
            s = scales[row, blk]
            block_orig = orig_weight[row, blk*32:(blk+1)*32]
            block_qdq = result[row, blk*32:(blk+1)*32]
            manual = np.clip(np.round(block_orig / s), -8, 7) * s
            if not np.allclose(manual, block_qdq, atol=1e-6):
                print(f"  MISMATCH at row={row}, blk={blk}!")
                print(f"    manual[:4]: {manual[:4]}")
                print(f"    aimet[:4]:  {block_qdq[:4]}")
    print("Block comparison done.")
    
except AttributeError as e:
    print(f"AttributeError: {e}")
    print("Trying alternative approach...")
    
    # Try calling through the quant_info
    try:
        qi = qtzr.quant_info
        print(f"\nquant_info attributes: {[a for a in dir(qi) if not a.startswith('__')]}")
    except Exception as e2:
        print(f"Error: {e2}")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
    # Alternative: try to get quantized weight from the model by running a pass
    print("\n\nAlternative: Running a single forward pass to extract quantized weight...")
    print("Creating a custom model that outputs the quantized weight directly...")
    
    # Modify model to add an output for the quantized weight
    quant_model = quant_sim.model.model
    qdq_output_name = weight_name + "_qdq"
    
    # Check if this tensor exists in the graph
    found = False
    for node in quant_model.graph.node:
        for out in node.output:
            if out == qdq_output_name:
                found = True
                break
    print(f"Found {qdq_output_name}: {found}")
