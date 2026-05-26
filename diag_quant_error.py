"""Diagnostic: Measure actual weight quantization error.
Manually apply the encoding to check if scales match the weights."""

import json
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'

# Load ONNX model with weights
print("Loading ONNX model with weights...")
m = onnx.load(f"{checkpoint}/model_seqlen128_cl256.onnx", load_external_data=False)
load_external_data_for_model(m, checkpoint)

# Build weight lookup
weights = {}
for init in m.graph.initializer:
    weights[init.name] = numpy_helper.to_array(init)

# Load encodings
with open(f"{checkpoint}/model.encodings", 'r') as f:
    enc = json.load(f)

param_encs = {item['name']: item for item in enc['param_encodings']}

# Check a few weights
test_weights = [
    "model.model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.model.layers.0.mlp.gate_proj.weight",
    "model.lm_head.weight",
    "model.model.layers.3.mlp.gate_proj.weight",  # full_attn layer MLP
]

for wname in test_weights:
    if wname not in weights:
        print(f"\n{wname}: NOT FOUND in ONNX model")
        continue
    if wname not in param_encs:
        print(f"\n{wname}: NOT FOUND in encodings")
        continue
    
    w = weights[wname]
    enc_info = param_encs[wname]
    
    scales = np.array(enc_info['scale'], dtype=np.float32)
    offsets = np.array(enc_info['offset'], dtype=np.float32)
    block_size = enc_info['block_size']
    bw = enc_info['bw']
    is_sym = enc_info['is_sym']
    
    print(f"\n{'='*70}")
    print(f"Weight: {wname}")
    print(f"  Shape: {w.shape}, dtype: {w.dtype}")
    print(f"  Values: min={w.min():.6f}, max={w.max():.6f}, std={w.std():.6f}")
    print(f"  Encoding: bw={bw}, block_size={block_size}, is_sym={is_sym}")
    print(f"  Num scales: {len(scales)}, scale range: [{scales.min():.8e}, {scales.max():.8e}]")
    
    # Expected number of blocks
    total_elements = w.size
    expected_blocks = total_elements // block_size
    print(f"  Total elements: {total_elements}, Expected blocks: {expected_blocks}")
    print(f"  Actual num scales: {len(scales)}")
    
    if len(scales) != expected_blocks:
        print(f"  *** MISMATCH: expected {expected_blocks} scales but got {len(scales)} ***")
        # Try to figure out the actual block layout
        if w.ndim == 2:
            # Try blocking along last axis
            blocks_last = w.shape[0] * (w.shape[1] // block_size)
            # Try blocking along first axis
            blocks_first = (w.shape[0] // block_size) * w.shape[1]
            print(f"  Blocks if along last axis (cols): {blocks_last}")
            print(f"  Blocks if along first axis (rows): {blocks_first}")
            if blocks_last == len(scales):
                print(f"  -> Scales match blocking along LAST axis (columns)")
            elif blocks_first == len(scales):
                print(f"  -> Scales match blocking along FIRST axis (rows)")
    
    # Simulate quantization manually
    # For per-block: reshape weight into blocks of block_size
    w_flat = w.flatten()
    
    if len(scales) == total_elements // block_size:
        # Simple flat blocking
        n_blocks = len(scales)
        w_blocked = w_flat[:n_blocks * block_size].reshape(n_blocks, block_size)
        
        # Quantize: q = round(w / scale) + offset (for asymmetric)
        # For symmetric INT4: range is [-8, 7], offset is -8
        if is_sym:
            q_min = -2**(bw-1)
            q_max = 2**(bw-1) - 1
        else:
            q_min = 0
            q_max = 2**bw - 1
        
        # Apply quantization per block
        scales_expanded = scales[:, np.newaxis]  # shape: (n_blocks, 1)
        offsets_expanded = offsets[:, np.newaxis]
        
        # quantize
        q_vals = np.clip(np.round(w_blocked / scales_expanded - offsets_expanded), q_min, q_max)
        # dequantize
        w_dequant = (q_vals + offsets_expanded) * scales_expanded
        
        # Compute error
        error = w_blocked - w_dequant
        rel_error = np.abs(error) / (np.abs(w_blocked) + 1e-10)
        
        print(f"\n  Quantization error (flat blocking):")
        print(f"    Absolute: mean={np.abs(error).mean():.6e}, max={np.abs(error).max():.6e}")
        print(f"    Relative: mean={rel_error.mean():.4f}, max={rel_error.max():.4f}")
        print(f"    SQNR (dB): {10*np.log10(np.mean(w_blocked**2) / (np.mean(error**2) + 1e-30)):.1f}")
    
    # Also try per-row blocking (if 2D weight)
    if w.ndim == 2:
        rows, cols = w.shape
        # Try blocking along columns (last axis)
        if cols % block_size == 0:
            n_blocks_per_row = cols // block_size
            total_blocks_last = rows * n_blocks_per_row
            if total_blocks_last == len(scales):
                w_reshaped = w.reshape(rows, n_blocks_per_row, block_size)
                scales_reshaped = scales.reshape(rows, n_blocks_per_row, 1)
                offsets_reshaped = offsets.reshape(rows, n_blocks_per_row, 1)
                
                if is_sym:
                    q_min = -2**(bw-1)
                    q_max = 2**(bw-1) - 1
                else:
                    q_min = 0
                    q_max = 2**bw - 1
                
                q_vals = np.clip(np.round(w_reshaped / scales_reshaped - offsets_reshaped), q_min, q_max)
                w_dequant = (q_vals + offsets_reshaped) * scales_reshaped
                
                error = w_reshaped - w_dequant
                print(f"\n  Quantization error (per-row, blocking along cols):")
                print(f"    Absolute: mean={np.abs(error).mean():.6e}, max={np.abs(error).max():.6e}")
                print(f"    SQNR (dB): {10*np.log10(np.mean(w_reshaped**2) / (np.mean(error**2) + 1e-30)):.1f}")
        
        # Try blocking along rows (first axis)
        if rows % block_size == 0:
            n_blocks_per_col = rows // block_size
            total_blocks_first = n_blocks_per_col * cols
            if total_blocks_first == len(scales):
                w_reshaped = w.reshape(n_blocks_per_col, block_size, cols).transpose(0, 2, 1)
                # This gives shape (n_blocks_per_col, cols, block_size)
                w_reshaped = w_reshaped.reshape(total_blocks_first, block_size)
                scales_expanded = scales[:, np.newaxis]
                offsets_expanded = offsets[:, np.newaxis]
                
                if is_sym:
                    q_min = -2**(bw-1)
                    q_max = 2**(bw-1) - 1
                else:
                    q_min = 0
                    q_max = 2**bw - 1
                
                q_vals = np.clip(np.round(w_reshaped / scales_expanded - offsets_expanded), q_min, q_max)
                w_dequant = (q_vals + offsets_expanded) * scales_expanded
                
                error = w_reshaped - w_dequant
                print(f"\n  Quantization error (blocking along rows/first axis):")
                print(f"    Absolute: mean={np.abs(error).mean():.6e}, max={np.abs(error).max():.6e}")
                print(f"    SQNR (dB): {10*np.log10(np.mean(w_reshaped**2) / (np.mean(error**2) + 1e-30)):.1f}")
