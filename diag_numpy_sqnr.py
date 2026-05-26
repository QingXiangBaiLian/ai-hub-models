"""Pure numpy per-layer weight SQNR analysis - no AIMET C++ needed.
Loads weights from ONNX and encodings from JSON, applies QDQ manually."""
import sys, json
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'

# Load ONNX model weights
print("Loading ONNX model...")
onnx_model = onnx.load(f'{checkpoint}/model_seqlen128_cl256.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, checkpoint)
init_dict = {t.name: numpy_helper.to_array(t) for t in onnx_model.graph.initializer}
print(f"  Loaded {len(init_dict)} initializers")

# Load encodings
print("Loading encodings...")
with open(f'{checkpoint}/model.encodings', 'r') as f:
    enc_data = json.load(f)

param_encodings_list = enc_data.get('param_encodings', [])
print(f"  {len(param_encodings_list)} param encodings")

# Build dict: name -> encoding
param_encodings = {e['name']: e for e in param_encodings_list}

# For each weight encoding, apply QDQ and compute SQNR
print(f"\n{'Weight Name':<70} {'Shape':<20} {'EncShape':<15} {'SQNR(dB)':<10} {'MaxDiff':<10}")
print("-" * 130)

results = []

for enc_name, enc in sorted(param_encodings.items()):
    # Find corresponding weight in initializer
    if enc_name not in init_dict:
        continue
    
    weight = init_dict[enc_name].astype(np.float32)
    
    # Parse encoding
    bitwidth = enc.get('bw', 4)
    is_symmetric = enc.get('is_sym', True)
    
    dtype = enc.get('dtype', 'INT')
    if dtype == 'FLOAT' or bitwidth == 16:
        # FP16 quantization - skip
        continue
    
    # Get scale and offset arrays
    scale = np.array(enc['scale'], dtype=np.float32)
    offset = np.array(enc['offset'], dtype=np.float32)
    
    # Determine quantization parameters
    qmin = -(2**(bitwidth-1))  # -8 for 4-bit
    qmax = 2**(bitwidth-1) - 1  # 7 for 4-bit
    
    # Determine block structure from encoding
    enc_type = enc.get('enc_type', 'PER_CHANNEL')
    block_size = enc.get('block_size', 0) or 0
    
    if block_size > 0:
        # Per-block quantization
        # Weight shape: (out, in) for MatMul or (out, in, 1, 1) for Conv1x1
        # channel_axis=0, block_axis=1, block_size=32
        
        if weight.ndim == 4:
            # Conv 1x1: (out, in, 1, 1) -> treat as (out, in)
            w_2d = weight.reshape(weight.shape[0], weight.shape[1])
        elif weight.ndim == 2:
            w_2d = weight
        else:
            # Skip unusual shapes
            print(f"{enc_name:<70} {str(weight.shape):<20} SKIP (ndim={weight.ndim})")
            continue
        
        num_channels = w_2d.shape[0]  # channel axis = 0
        in_features = w_2d.shape[1]   # block axis = 1
        num_blocks = in_features // block_size
        
        if num_blocks * block_size != in_features:
            print(f"{enc_name:<70} {str(weight.shape):<20} SKIP (not divisible)")
            continue
        
        # Expected scale shape: (num_channels * num_blocks,) flattened from (num_channels, num_blocks)
        expected_num_scales = num_channels * num_blocks
        
        if len(scale) != expected_num_scales:
            print(f"{enc_name:<70} {str(weight.shape):<20} SCALE MISMATCH (got {len(scale)}, expected {expected_num_scales})")
            results.append((enc_name, -99, 0, weight.shape, f"scale_mismatch_{len(scale)}_vs_{expected_num_scales}"))
            continue
        
        # Reshape weight into blocks: (num_channels, num_blocks, block_size)
        w_blocks = w_2d.reshape(num_channels, num_blocks, block_size)
        
        # Reshape scales: (num_channels * num_blocks) -> (num_channels, num_blocks, 1)
        # The encodings are stored in C-order: channel varies slowest, block varies fastest
        scale_2d = scale.reshape(num_channels, num_blocks, 1)
        offset_2d = offset.reshape(num_channels, num_blocks, 1)
        
        # Apply QDQ: quantize then dequantize
        # AIMET formula: q = clip(round(x / scale) - offset, 0, 2^bw-1)
        #                x_hat = (q + offset) * scale
        # With offset = -8 for signed symmetric: effectively signed quantization
        # Simplified for symmetric: x_hat = clip(round(x/scale), -8, 7) * scale
        
        # Using AIMET's unsigned convention:
        # q_unsigned = clip(round(x / scale - offset), 0, 15)  [offset is -8]
        # x_hat = (q_unsigned + offset) * scale
        quantized = np.clip(np.round(w_blocks / scale_2d - offset_2d), 0, 2**bitwidth - 1)
        dequantized = (quantized + offset_2d) * scale_2d
        
        # Reshape back
        dequantized = dequantized.reshape(w_2d.shape)
        if weight.ndim == 4:
            dequantized = dequantized.reshape(weight.shape)
    
    elif enc_type == 'PER_CHANNEL':
        # Per-channel (conv1d with small channel dim)
        if weight.ndim == 3:
            # (out, in, kernel) - channel_axis=0
            num_channels = weight.shape[0]
            if len(scale) != num_channels:
                print(f"{enc_name:<70} {str(weight.shape):<20} SCALE MISMATCH")
                continue
            scale_bc = scale.reshape(-1, 1, 1)
            offset_bc = offset.reshape(-1, 1, 1)
        elif weight.ndim == 2:
            num_channels = weight.shape[0]
            if len(scale) != num_channels:
                print(f"{enc_name:<70} {str(weight.shape):<20} SCALE MISMATCH")
                continue
            scale_bc = scale.reshape(-1, 1)
            offset_bc = offset.reshape(-1, 1)
        else:
            print(f"{enc_name:<70} {str(weight.shape):<20} SKIP")
            continue
        
        quantized = np.clip(np.round(weight / scale_bc - offset_bc), 0, 2**bitwidth - 1)
        dequantized = (quantized + offset_bc) * scale_bc
    
    else:
        # Per-tensor
        quantized = np.clip(np.round(weight / scale[0] - offset[0]), 0, 2**bitwidth - 1)
        dequantized = (quantized + offset[0]) * scale[0]
    
    # Compute SQNR
    error = weight - dequantized
    mse = np.mean(error**2)
    signal = np.mean(weight**2)
    
    if mse > 0 and signal > 0:
        sqnr = 10 * np.log10(signal / mse)
    elif mse == 0:
        sqnr = float('inf')
    else:
        sqnr = -float('inf')
    
    max_diff = np.max(np.abs(error))
    
    enc_shape_str = f"{len(scale)}"
    if block_size:
        enc_shape_str += f"(bs={block_size})"
    
    results.append((enc_name, sqnr, max_diff, weight.shape, ""))
    
    # Print layers with SQNR < 15 dB
    if sqnr < 15:
        flag = " *** BAD" if sqnr < 10 else ""
        print(f"{enc_name:<70} {str(weight.shape):<20} {enc_shape_str:<15} {sqnr:<10.2f} {max_diff:<10.6f}{flag}")

# Summary
print(f"\n{'='*100}")
sqnr_values = [r[1] for r in results if r[1] != float('inf') and r[1] != -float('inf') and r[1] > -90]
print(f"Total weights analyzed: {len(results)}")
print(f"SQNR: min={min(sqnr_values):.2f}, max={max(sqnr_values):.2f}, mean={np.mean(sqnr_values):.2f}, median={np.median(sqnr_values):.2f}")
print(f"Layers with SQNR < 10 dB: {sum(1 for r in results if r[1] < 10)}")
print(f"Layers with SQNR < 15 dB: {sum(1 for r in results if r[1] < 15)}")
print(f"Layers with SQNR < 20 dB: {sum(1 for r in results if r[1] < 20)}")

# Print worst 15 layers
print(f"\nWorst 15 layers:")
for name, sqnr, md, shape, note in sorted(results, key=lambda x: x[1])[:15]:
    print(f"  {name:<60} shape={str(shape):<20} SQNR={sqnr:.2f} dB {note}")

# Print best 5 layers
print(f"\nBest 5 layers:")
valid = [r for r in results if r[1] != float('inf')]
for name, sqnr, md, shape, note in sorted(valid, key=lambda x: -x[1])[:5]:
    print(f"  {name:<60} shape={str(shape):<20} SQNR={sqnr:.2f} dB")
