"""Check what node types are in the QuantSim ONNX graph."""
import sys
sys.path.insert(0, '/workspace/ai-hub-models/src')
import os, onnx, logging
import numpy as np
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_model
import torch
logging.getLogger('Quant').setLevel(logging.ERROR)

checkpoint = '/workspace/qwen3_5_0_8b_w4a16_static'
onnx_model = onnx.load(f'{checkpoint}/model_seqlen128_cl256.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, checkpoint)

from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx
from qai_hub_models.utils.base_model import Precision
from aimet_onnx.quantsim import load_encodings_to_sim

quant_sim = Qwen3_5_0_8B_AIMETOnnx.create_quantsim(onnx_model, torch.device('cpu'), Precision.w4a16)
load_encodings_to_sim(quant_sim, f'{checkpoint}/model.encodings', strict=False)

# Check node types in QuantSim model
from collections import Counter
model = quant_sim.model.model
op_types = Counter(n.op_type for n in model.graph.node)
print('Node type counts:')
for op, count in op_types.most_common(30):
    print(f'  {op}: {count}')

# Find custom/quantize ops
quant_ops = [n for n in model.graph.node if 'quant' in n.op_type.lower() or 'Qc' in n.op_type or n.domain != '']
print(f'\nNodes with non-empty domain or quant in name: {len(quant_ops)}')
if quant_ops:
    # Show unique types
    qtypes = Counter((n.op_type, n.domain) for n in quant_ops)
    for (ot, dom), cnt in qtypes.most_common():
        print(f'  {ot} (domain={dom}): {cnt}')
    
    # Show details of first one
    first = quant_ops[0]
    print(f'\nFirst quant op details:')
    print(f'  op_type: {first.op_type}')
    print(f'  domain: {first.domain}')
    print(f'  name: {first.name}')
    print(f'  inputs: {list(first.input)[:5]}')
    print(f'  outputs: {list(first.output)[:5]}')
    for attr in first.attribute:
        if attr.type == 2:  # INT
            print(f'  attr {attr.name} (int): {attr.i}')
        elif attr.type == 1:  # FLOAT
            print(f'  attr {attr.name} (float): {attr.f}')
        elif attr.type == 3:  # STRING
            print(f'  attr {attr.name} (string): {attr.s}')
        elif attr.type == 7:  # INTS
            print(f'  attr {attr.name} (ints): {list(attr.ints)[:10]}')

# Look for gate_proj weight related nodes
print('\n\nNodes using gate_proj weight:')
target = 'model.model.layers.0.mlp.gate_proj.weight'
for n in model.graph.node:
    if target in n.input or (target + '_qdq') in n.input:
        print(f'  {n.op_type} (domain={n.domain}): inputs={list(n.input)[:4]}... outputs={list(n.output)[:2]}')
        for attr in n.attribute:
            if attr.type == 2:
                print(f'    {attr.name}: {attr.i}')
