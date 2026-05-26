"""Diagnostic: Verify quantization error with CORRECT AIMET formula.
AIMET symmetric INT4 uses unsigned representation [0, 15] with offset=-8."""

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

# Test with correct AIMET formula
wname = "model.model.layers.0.mlp.gate_proj.weight"
w = weights[wname]
enc_info = param_encs[wname]

scales = np.array(enc_info['scale'], dtype=np.float64)
offsets = np.array(enc_info['offset'], dtype=np.float64)
block_size = enc_info['block_size']
bw = enc_info['bw']

print(f"Weight: {wname}")
print(f"  Shape: {w.shape}")
print(f"  Values: min={w.min():.6f}, max={w.max():.6f}, std={w.std():.6f}")
print(f"  Scales: min={scales.min():.6e}, max={scales.max():.6e}, mean={scales.mean():.6e}")
print(f"  Offsets: unique values = {np.unique(offsets)}")

# Method 1: WRONG formula (what I used before - signed range with offset in clamp bounds)
print(f"\n--- Method 1: WRONG formula (signed [-8,7] range) ---")
w_flat = w.flatten().astype(np.float64)
n_blocks = len(scales)
w_blocked = w_flat[:n_blocks * block_size].reshape(n_blocks, block_size)
scales_exp = scales[:, np.newaxis]
offsets_exp = offsets[:, np.newaxis]

# Wrong: clip to [-8, 7]
q_vals = np.clip(np.round(w_blocked / scales_exp - offsets_exp), -8, 7)
w_dequant = (q_vals + offsets_exp) * scales_exp
error = w_blocked - w_dequant
sqnr = 10*np.log10(np.mean(w_blocked**2) / (np.mean(error**2) + 1e-30))
print(f"  SQNR: {sqnr:.1f} dB")
print(f"  Mean abs error: {np.abs(error).mean():.6e}")

# Method 2: CORRECT formula (unsigned [0, 15] range, offset as zero-point shift)
print(f"\n--- Method 2: Unsigned [0, 15] with offset as zero-point ---")
# q = clip(round(w/scale) - offset, 0, 2^bw - 1)
# w_hat = (q + offset) * scale
q_max = 2**bw - 1  # 15
q_vals2 = np.clip(np.round(w_blocked / scales_exp) - offsets_exp, 0, q_max)
w_dequant2 = (q_vals2 + offsets_exp) * scales_exp
error2 = w_blocked - w_dequant2
sqnr2 = 10*np.log10(np.mean(w_blocked**2) / (np.mean(error2**2) + 1e-30))
print(f"  SQNR: {sqnr2:.1f} dB")
print(f"  Mean abs error: {np.abs(error2).mean():.6e}")

# Method 3: Simple symmetric (q in [-8, 7], no offset in formula)
print(f"\n--- Method 3: Simple symmetric [-8, 7], ignore offset ---")
q_vals3 = np.clip(np.round(w_blocked / scales_exp), -8, 7)
w_dequant3 = q_vals3 * scales_exp
error3 = w_blocked - w_dequant3
sqnr3 = 10*np.log10(np.mean(w_blocked**2) / (np.mean(error3**2) + 1e-30))
print(f"  SQNR: {sqnr3:.1f} dB")
print(f"  Mean abs error: {np.abs(error3).mean():.6e}")

# Method 4: Symmetric with half-range [-7, 7]
print(f"\n--- Method 4: Symmetric [-7, 7] (restricted symmetric) ---")
q_vals4 = np.clip(np.round(w_blocked / scales_exp), -7, 7)
w_dequant4 = q_vals4 * scales_exp
error4 = w_blocked - w_dequant4
sqnr4 = 10*np.log10(np.mean(w_blocked**2) / (np.mean(error4**2) + 1e-30))
print(f"  SQNR: {sqnr4:.1f} dB")
print(f"  Mean abs error: {np.abs(error4).mean():.6e}")

# Check: What if scale should be computed as max_abs / 7 for this block?
print(f"\n--- Ideal scales check ---")
# For each block, compute what the ideal scale would be
w_blocked_abs_max = np.abs(w_blocked).max(axis=1)  # max per block
ideal_scales = w_blocked_abs_max / 7.0  # For symmetric [-7, 7]
ratio = scales.flatten() / (ideal_scales + 1e-30)
print(f"  Ratio of actual/ideal scale: mean={ratio.mean():.4f}, std={ratio.std():.4f}")
print(f"  Ratio min={ratio.min():.4f}, max={ratio.max():.4f}")
print(f"  % of blocks where actual scale > 2x ideal: {(ratio > 2).mean()*100:.1f}%")
print(f"  % of blocks where actual scale < 0.5x ideal: {(ratio < 0.5).mean()*100:.1f}%")

# For symmetric [-8, 7], ideal scale = max_abs / 8 (for the negative side)
ideal_scales_8 = w_blocked_abs_max / 8.0
ratio_8 = scales.flatten() / (ideal_scales_8 + 1e-30)
print(f"\n  If range is [-8, 7], ideal scale = max_abs/8:")
print(f"  Ratio actual/ideal: mean={ratio_8.mean():.4f}, std={ratio_8.std():.4f}")

# What does AIMET actually use for min_max symmetric?
# In AIMET min_max scheme: scale = (max - min) / (2^bw - 1)
# For symmetric: max_abs_val / (2^(bw-1) - 1) OR max_abs_val / 2^(bw-1)
# Let's check both
ideal_a = w_blocked_abs_max / (2**(bw-1))  # max_abs / 8
ideal_b = w_blocked_abs_max / (2**(bw-1) - 1)  # max_abs / 7

print(f"\n  Comparison of actual scales vs ideals (first 5 blocks):")
for i in range(5):
    print(f"    Block {i}: actual={scales[i]:.6e}, max_abs={w_blocked_abs_max[i]:.6e}, "
          f"ideal_/8={ideal_a[i]:.6e}, ideal_/7={ideal_b[i]:.6e}")
