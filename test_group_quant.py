"""Test per-group (block) quantization with group_size=64."""
import numpy as np
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')

from aimet_onnx.quantsim import QuantizationSimModel, compute_encodings
import onnx
import os

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'
onnx_path = os.path.join(checkpoint, 'model_seqlen1_cl256.onnx')
encodings_path = os.path.join(checkpoint, 'model.encodings')

print('Loading ONNX model...')
model_proto = onnx.load(onnx_path, load_external_data=True)
print('Creating QuantizationSimModel...')
quant_sim = QuantizationSimModel(model=model_proto, path=checkpoint)
print('Loading encodings...')
quant_sim.load_encodings(encodings_path, strict=False)
print(f'Loaded. Total quantizers: {len(quant_sim.qc_quantize_op_dict)}')

# Check weight quantizer params
print('\nWeight quantizer info (first 5):')
count = 0
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names:
        continue
    if not qc_op.enabled:
        continue
    if count < 5:
        tqp = qc_op.tensor_quantizer_params
        qi = qc_op.quant_info
        print(f'  {name}:')
        print(f'    bitwidth={qc_op.bitwidth}')
        print(f'    usePerChannelMode={qi.usePerChannelMode}')
        print(f'    channelAxis={qi.channelAxis}')
        print(f'    blockAxis={qi.blockAxis}')
        print(f'    blockSize={qi.blockSize}')
        if tqp:
            print(f'    tensor_shape={tqp.tensor_shape}')
            print(f'    channel_axis={tqp.channel_axis}')
            print(f'    block_axis={tqp.block_axis}')
        enc = qc_op.encodings
        if enc:
            print(f'    num_encodings={len(enc)}')
        count += 1

# Try enabling blockwise quantization on one weight quantizer
print('\n--- Testing block quantization with group_size=64 ---')
count = 0
test_op = None
test_name = None
for name, qc_op in quant_sim.qc_quantize_op_dict.items():
    if name in quant_sim.activation_names:
        continue
    if not qc_op.enabled:
        continue
    if count == 0:
        test_op = qc_op
        test_name = name
    count += 1

if test_op:
    tqp = test_op.tensor_quantizer_params
    qi = test_op.quant_info
    print(f'\nTest op: {test_name}')
    print(f'  tensor_shape={tqp.tensor_shape if tqp else None}')
    print(f'  block_axis={tqp.block_axis if tqp else None}')
    
    # Check if the input dim is divisible by 64
    if tqp and tqp.tensor_shape:
        block_axis = tqp.block_axis
        if block_axis is not None:
            dim = tqp.tensor_shape[block_axis]
            print(f'  dim at block_axis={block_axis}: {dim}')
            print(f'  divisible by 64: {dim % 64 == 0}')
            
            # Try enabling blockwise
            try:
                test_op.set_bitwidth(8)
                test_op._enable_blockwise_quantization(64)
                print(f'  SUCCESS: blockwise enabled with group_size=64')
                print(f'  new blockSize={test_op.quant_info.blockSize}')
            except Exception as e:
                print(f'  FAILED: {e}')
        else:
            print(f'  block_axis is None - cannot enable blockwise')

print('\nDone!')
