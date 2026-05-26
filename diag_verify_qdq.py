"""Verify weight QDQ for a specific layer in the full model."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import numpy as np
import onnx, logging
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model
import torch
logging.getLogger('Quant').setLevel(logging.ERROR)

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'
onnx_model = onnx.load(f'{checkpoint}/model_seqlen128_cl256.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, checkpoint)

print("Creating QuantSim...")
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision
from aimet_onnx.quantsim import load_encodings_to_sim

quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(onnx_model, torch.device('cpu'), Precision.w4a16)
load_encodings_to_sim(quant_sim, f'{checkpoint}/model.encodings', strict=False)

# Test multiple weight types
test_weights = [
    'model.model.layers.0.mlp.gate_proj.weight',  # MatMul [3584, 1024]
    'model.model.layers.0.mlp.down_proj.weight',   # Conv 1x1 [1024, 3584, 1, 1]
    'model.model.layers.0.linear_attn.conv1d.weight',  # Conv1d [6144, 1, 4]
    'model.model.layers.0.linear_attn.in_proj_qkv.weight',  # MatMul [6144, 1024]
]

# Get weights from ONNX model
init_dict = {t.name: numpy_helper.to_array(t) for t in onnx_model.graph.initializer}

for weight_name in test_weights:
    if weight_name not in init_dict:
        print(f"\n{weight_name}: NOT FOUND in initializers")
        continue
    
    weight = init_dict[weight_name]
    qtzr = quant_sim.qc_quantize_op_dict.get(weight_name)
    
    if qtzr is None:
        print(f"\n{weight_name}: NO QUANTIZER")
        continue
    
    print(f"\n{'='*80}")
    print(f"Weight: {weight_name}")
    print(f"  Shape: {weight.shape}")
    print(f"  Quantizer enabled: {qtzr.enabled}")
    print(f"  Bitwidth: {qtzr.bitwidth}")
    print(f"  Per-channel: {qtzr.quant_info.usePerChannelMode}")
    print(f"  Channel axis: {qtzr.quant_info.channelAxis}")
    print(f"  Block axis: {qtzr.quant_info.blockAxis}")
    print(f"  Block size: {qtzr.quant_info.blockSize}")
    print(f"  Op mode: {qtzr.quant_info.opMode}")
    
    if qtzr.tensor_quantizer_params:
        print(f"  TQP tensor_shape: {qtzr.tensor_quantizer_params.tensor_shape}")
        print(f"  TQP channel_axis: {qtzr.tensor_quantizer_params.channel_axis}")
        print(f"  TQP block_axis: {qtzr.tensor_quantizer_params.block_axis}")
    
    # Get encodings
    encs = qtzr.quant_info.tensorQuantizerRef.getEncodings()
    print(f"  Num TfEncodings: {len(encs)}")
    if len(encs) > 0:
        print(f"  First enc: delta={encs[0].delta:.8f}, offset={encs[0].offset:.1f}, bw={encs[0].bw}")
        print(f"  Last enc: delta={encs[-1].delta:.8f}, offset={encs[-1].offset:.1f}, bw={encs[-1].bw}")
    
    # Now manually quantize-dequantize and compare with AIMET
    # AIMET's quantizeDequantize on the weight directly
    weight_flat = weight.flatten()
    
    # Call AIMET's quantizeDequantize
    import libpymo
    weight_tensor = torch.from_numpy(weight.copy())
    qdq_output = qtzr.quant_info.tensorQuantizerRef.quantizeDequantize(weight_tensor)
    qdq_output = qdq_output.numpy()
    
    # Manual QDQ for first few elements
    bw = qtzr.bitwidth
    block_size = qtzr.quant_info.blockSize
    ch_axis = qtzr.quant_info.channelAxis
    blk_axis = qtzr.quant_info.blockAxis
    
    # For the first channel, first block
    if block_size > 0 and len(weight.shape) >= 2:
        # Extract first block of weights
        if len(weight.shape) == 2:  # MatMul weight [out, in]
            block_weights = weight[0, :block_size]  # first channel, first block
            enc_idx = 0  # channel 0, block 0
        elif len(weight.shape) == 4:  # Conv weight [out, in, h, w]
            block_weights = weight[0, :block_size, 0, 0]  # first channel, first block
            enc_idx = 0  # channel 0, block 0
        else:  # 3D: Conv1d [out, in, k]
            block_weights = weight[0, :block_size, :]  # might not work for [6144,1,4]
            enc_idx = 0
        
        enc = encs[enc_idx]
        # Manual: round(w/scale) + offset, then clamp, then (q - offset) * scale
        # Note: offset is in UNSIGNED form here (offset = -8 for symmetric int4)
        scale = enc.delta
        offset = enc.offset  # -8 for symmetric int4
        qmin = 0  # unsigned
        qmax = 2**bw - 1  # 15 for int4
        
        q = np.round(block_weights / scale) - offset
        q = np.clip(q, qmin, qmax)
        manual_qdq = (q + offset) * scale
        
        # Compare with AIMET output for same positions
        if len(weight.shape) == 2:
            aimet_block = qdq_output[0, :block_size]
        elif len(weight.shape) == 4:
            aimet_block = qdq_output[0, :block_size, 0, 0]
        else:
            aimet_block = qdq_output[0, 0, :]  # For [6144,1,4], first channel
        
        print(f"\n  --- Block 0, Channel 0 comparison ---")
        print(f"  Original weights[:4]: {block_weights[:4]}")
        print(f"  Manual QDQ[:4]:       {manual_qdq[:4]}")
        print(f"  AIMET QDQ[:4]:        {aimet_block[:4]}")
        print(f"  Manual vs AIMET max diff: {np.max(np.abs(manual_qdq[:len(aimet_block)] - aimet_block[:len(manual_qdq)])):.8f}")
    
    # Overall QDQ quality
    mse = np.mean((weight - qdq_output)**2)
    signal = np.mean(weight**2)
    sqnr = 10 * np.log10(signal / mse) if mse > 0 else float('inf')
    print(f"\n  Overall weight SQNR: {sqnr:.2f} dB")
    print(f"  Weight range: [{weight.min():.4f}, {weight.max():.4f}]")
    print(f"  QDQ range: [{qdq_output.min():.4f}, {qdq_output.max():.4f}]")
    max_diff = np.max(np.abs(weight - qdq_output))
    print(f"  Max abs diff: {max_diff:.6f}")
