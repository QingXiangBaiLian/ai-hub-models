"""Verify weight QDQ for specific layers - check per-layer SQNR."""
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

# Get weights from ONNX model
init_dict = {t.name: numpy_helper.to_array(t) for t in onnx_model.graph.initializer}

# Test ALL weight quantizers - compute per-weight SQNR
print("\nPer-weight SQNR analysis:")
print(f"{'Weight Name':<60} {'Shape':<25} {'SQNR(dB)':<10} {'Max Diff':<10}")
print("-" * 110)

sqnr_list = []
bad_layers = []

for weight_name in sorted(quant_sim.param_names):
    qtzr = quant_sim.qc_quantize_op_dict.get(weight_name)
    if qtzr is None or not qtzr.enabled:
        continue
    if weight_name not in init_dict:
        continue
    
    weight = init_dict[weight_name]
    weight_tensor = torch.from_numpy(weight.copy())
    
    # Call AIMET's quantizeDequantize
    qdq_output = qtzr.quant_info.tensorQuantizerRef.quantizeDequantize(weight_tensor)
    qdq_output = qdq_output.numpy()
    
    # Compute SQNR
    mse = np.mean((weight - qdq_output)**2)
    signal = np.mean(weight**2)
    if mse > 0 and signal > 0:
        sqnr = 10 * np.log10(signal / mse)
    elif mse == 0:
        sqnr = float('inf')
    else:
        sqnr = -float('inf')
    
    max_diff = np.max(np.abs(weight - qdq_output))
    sqnr_list.append((weight_name, sqnr, max_diff, weight.shape))
    
    if sqnr < 10:  # Flag layers with bad SQNR
        bad_layers.append((weight_name, sqnr, max_diff, weight.shape))
        print(f"{weight_name:<60} {str(weight.shape):<25} {sqnr:<10.2f} {max_diff:<10.6f} *** BAD")
    elif sqnr < 15:
        print(f"{weight_name:<60} {str(weight.shape):<25} {sqnr:<10.2f} {max_diff:<10.6f}")

# Summary statistics
sqnr_values = [s[1] for s in sqnr_list if s[1] != float('inf')]
print(f"\n{'='*80}")
print(f"Total weight quantizers tested: {len(sqnr_list)}")
print(f"SQNR statistics: min={min(sqnr_values):.2f}, max={max(sqnr_values):.2f}, mean={np.mean(sqnr_values):.2f}, median={np.median(sqnr_values):.2f}")
print(f"Layers with SQNR < 10 dB: {len(bad_layers)}")
print(f"Layers with SQNR < 15 dB: {sum(1 for _, s, _, _ in sqnr_list if s < 15)}")
print(f"Layers with SQNR < 20 dB: {sum(1 for _, s, _, _ in sqnr_list if s < 20)}")

if bad_layers:
    print(f"\nWorst 10 layers:")
    for name, sqnr, md, shape in sorted(bad_layers, key=lambda x: x[1])[:10]:
        print(f"  {name}: SQNR={sqnr:.2f} dB, shape={shape}")
