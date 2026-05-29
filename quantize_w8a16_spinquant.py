"""
Quantize Qwen3.5-0.8B with w8a16 precision + SpinQuant on CPU.
Bypasses the CUDA check since SpinQuant is numpy-based.
"""
import sys
import os
import gc

sys.path.insert(0, '/workspace/ai-hub-models/src')
os.environ['PYTHONUNBUFFERED'] = '1'

import torch

from qai_hub_models.models.qwen3_5_0_8b import FP_Model, Model
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader

OUTPUT_DIR = '/workspace/qwen3_5_0_8b_w8a16_spinquant'
CONTEXT_LENGTH = 256
SEQ_LEN = 128
NUM_SAMPLES = 20
SPIN_QUANT_NUM_ITERATIONS = 200

print("=" * 60)
print("Qwen3.5-0.8B W8A16 + SpinQuant Quantization")
print("=" * 60)
print(f"Output dir: {OUTPUT_DIR}")
print(f"Context length: {CONTEXT_LENGTH}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Num samples: {NUM_SAMPLES}")
print(f"SpinQuant iterations: {SPIN_QUANT_NUM_ITERATIONS}")
print()

# Step 1: Create FP model
print("Step 1: Creating FP model...")
fp_model = FP_Model.from_pretrained(
    sequence_length=SEQ_LEN,
    context_length=CONTEXT_LENGTH,
).to(torch.device("cpu")).eval()
print("FP model created.")
print()

# Step 2: Create quantized model (w8a16)
print("Step 2: Creating quantized model (w8a16)...")
model_quant = Model.from_pretrained(
    context_length=CONTEXT_LENGTH,
    sequence_length=SEQ_LEN,
    precision=Precision.w8a16,
    checkpoint=None,
    host_device=torch.device("cpu"),
    fp_model=fp_model,
    _skip_quantsim_creation=False,
)
print("Quantized model created.")
print()

# Step 3: Get calibration data
print("Step 3: Getting calibration data...")
calib_data = model_quant.get_calibration_data(num_samples=NUM_SAMPLES)
assert calib_data is not None
dataloader = dataset_entries_to_dataloader(calib_data)
print(f"Calibration data ready: {len(dataloader)} batches")
print()

gc.collect()

# Step 4: Run quantization with SpinQuant (bypasses CLI CUDA check)
print("Step 4: Running quantization with SpinQuant...")
print("NOTE: SpinQuant on CPU may take a while but works (numpy-based).")
print()
model_quant.quantize(
    data=dataloader,
    num_samples=NUM_SAMPLES,
    use_spin_quant=True,
    spin_quant_num_iterations=SPIN_QUANT_NUM_ITERATIONS,
)
print()
print("Quantization complete!")
print()

# Step 5: Save checkpoint
print(f"Step 5: Saving checkpoint to {OUTPUT_DIR}...")
model_quant.save_calibrated_checkpoint(OUTPUT_DIR, fp_model=fp_model)
print("Checkpoint saved.")
print()

# Cleanup
model_quant = model_quant.to("cpu")
del model_quant
fp_model = fp_model.to("cpu")
del fp_model
gc.collect()

print("=" * 60)
print("Quantization DONE. Checkpoint at:", OUTPUT_DIR)
print("=" * 60)
