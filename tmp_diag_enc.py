"""Check if activation quantizers have proper encodings loaded."""
import sys
sys.path.insert(0, "/workspace/ai-hub-models/src")
import torch
from qai_hub_models.models.qwen3_5_0_8b import Model
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B as FP_Model

print("Loading quantized model...")
quant_model = Model.from_pretrained(
    checkpoint="/workspace/qwen3_5_0_8b_w4a16_static",
    sequence_length=128,
    context_length=256,
    host_device=torch.device("cpu"),
).to("cpu")

quant_sim = quant_model.quant_sim

# Check activation quantizers
act_names = quant_sim.activation_names
enabled_acts = []
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in act_names and qc_op.enabled:
        enabled_acts.append((name, qc_op))

print(f"Total enabled activation quantizers: {len(enabled_acts)}")

# Check encoding state
no_encoding = []
has_encoding = []
for name, qc_op in enabled_acts:
    # Check if encoding is set
    enc = qc_op.encodings
    if enc is None or (hasattr(enc, '__len__') and len(enc) == 0):
        no_encoding.append((name, qc_op.bitwidth))
    else:
        has_encoding.append((name, qc_op.bitwidth, enc))

print(f"  With encoding: {len(has_encoding)}")
print(f"  WITHOUT encoding: {len(no_encoding)}")

if no_encoding:
    print(f"\n=== Quantizers WITHOUT encodings (first 20): ===")
    for name, bw in no_encoding[:20]:
        print(f"  {name} (bw={bw})")
    if len(no_encoding) > 20:
        print(f"  ... and {len(no_encoding) - 20} more")

# Check a few with encoding to see if scales look reasonable
if has_encoding:
    print(f"\n=== Sample quantizers WITH encodings: ===")
    for name, bw, enc in has_encoding[:10]:
        if hasattr(enc, 'scale'):
            print(f"  {name}: bw={bw}, scale={enc.scale}")
        elif isinstance(enc, list) and len(enc) > 0:
            e = enc[0]
            if hasattr(e, 'scale'):
                print(f"  {name}: bw={bw}, scale={e.scale}")
            else:
                print(f"  {name}: bw={bw}, enc_type={type(e)}, enc={str(e)[:100]}")
        else:
            print(f"  {name}: bw={bw}, enc_type={type(enc)}, repr={str(enc)[:100]}")

# Check op_mode (passthrough vs quantize-dequantize)
from collections import Counter
modes = Counter()
bws = Counter()
for name, qc_op in enabled_acts:
    modes[str(qc_op.op_mode)] += 1
    bws[qc_op.bitwidth] += 1

print(f"\n=== Op modes: ===")
for mode, count in modes.most_common():
    print(f"  {mode}: {count}")

print(f"\n=== Bitwidth distribution: ===")
for bw, count in bws.most_common():
    print(f"  {bw}-bit: {count}")

# Check if any quantizers are in "updateStats" mode (not quantizing, just collecting)
update_stats_count = 0
for name, qc_op in enabled_acts:
    if hasattr(qc_op, 'op_mode'):
        mode_val = qc_op.op_mode
        # OpMode enum: 0=updateStats, 1=oneShotQuantizeDequantize, 2=quantizeDequantize, 3=passThrough
        if hasattr(mode_val, 'value') and mode_val.value == 0:
            update_stats_count += 1
        elif str(mode_val) == 'OpMode.updateStats' or 'update' in str(mode_val).lower():
            update_stats_count += 1

print(f"\n  Quantizers in updateStats mode: {update_stats_count}")
print("\nDONE")
