"""Quick check of weight quantizer params via the model loading pipeline."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')

# Monkey-patch _convert_weights_to_int8 to just inspect instead of convert
import qai_hub_models.models._shared.qwen3_5.model as qwen_model

original_convert = qwen_model.Qwen3_5Base_AIMETOnnx._convert_weights_to_int8

def inspect_and_skip(self):
    """Inspect weight quantizer params without converting."""
    if self.quant_sim is None:
        return
    
    count = 0
    divisible_64 = 0
    not_divisible = 0
    no_block_axis = 0
    
    for name, qc_op in self.quant_sim.qc_quantize_op_dict.items():
        if name in self.quant_sim.activation_names:
            continue
        if not qc_op.enabled:
            continue
        if qc_op.bitwidth != 4:
            continue
        
        tqp = qc_op.tensor_quantizer_params
        qi = qc_op.quant_info
        
        if count < 5:
            print(f'  {name}:')
            print(f'    bw={qc_op.bitwidth}, perChannel={qi.usePerChannelMode}')
            print(f'    channelAxis={qi.channelAxis}, blockAxis={qi.blockAxis}, blockSize={qi.blockSize}')
            if tqp:
                print(f'    tensor_shape={tqp.tensor_shape}')
                print(f'    channel_axis={tqp.channel_axis}, block_axis={tqp.block_axis}')
        
        # Check divisibility by 64
        if tqp and tqp.block_axis is not None:
            dim = tqp.tensor_shape[tqp.block_axis]
            if dim % 64 == 0:
                divisible_64 += 1
            else:
                not_divisible += 1
                if count < 10:
                    print(f'  NOT DIVISIBLE: {name}, dim={dim} at axis={tqp.block_axis}')
        else:
            no_block_axis += 1
            if count < 10 and tqp:
                print(f'  NO BLOCK AXIS: {name}, shape={tqp.tensor_shape}')
        
        count += 1
    
    print(f'\nTotal INT4 weight quantizers: {count}')
    print(f'  Divisible by 64: {divisible_64}')
    print(f'  NOT divisible by 64: {not_divisible}')
    print(f'  No block_axis: {no_block_axis}')
    
    # Don't actually convert - just exit after inspection
    import os
    os._exit(0)

qwen_model.Qwen3_5Base_AIMETOnnx._convert_weights_to_int8 = inspect_and_skip

# Now load the model through the normal pipeline
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
print('Loading model...')
model = Qwen3_5_0_8B_AIMETOnnx.from_pretrained(
    checkpoint='/workspace/qwen3_5_0_8b_w4a16_static',
    sequence_length=128,
    context_length=256,
)
