import onnx
from onnx.external_data_helper import load_external_data_for_model

model = onnx.load('/workspace/qwen3_5_0_8b_w4a16_static/model_seqlen128_cl256.onnx', load_external_data=False)

# Find Conv nodes and their weight shapes
conv_weights = {}
for node in model.graph.node:
    if node.op_type == 'Conv':
        weight_name = node.input[1]
        conv_weights[node.name] = weight_name

# Get shapes from initializers
init_shapes = {t.name: list(t.dims) for t in model.graph.initializer}
print('Conv layers and weight shapes:')
for node_name, w_name in sorted(conv_weights.items()):
    shape = init_shapes.get(w_name, 'NOT IN INITIALIZER')
    divisible = 'YES' if isinstance(shape, list) and len(shape) >= 2 and shape[1] % 32 == 0 else 'NO'
    print(f'  {w_name}: {shape} (block32 compatible: {divisible})')
print(f'Total Conv nodes: {len(conv_weights)}')

# Also check MatMul nodes with transposed_params
matmul_info = []
for node in model.graph.node:
    if node.op_type == 'MatMul':
        inp1 = node.input[1]
        # Check if inp1 comes from a Transpose
        transposed = False
        for other_node in model.graph.node:
            if inp1 in other_node.output and other_node.op_type == 'Transpose':
                transposed = True
                inp1 = other_node.input[0]  # original weight name
                break
        shape = init_shapes.get(inp1)
        if shape:
            matmul_info.append((node.name, inp1, shape, transposed))

print(f'\nMatMul nodes with weights: {len(matmul_info)}')
# Check how many have transposed vs not
transposed_count = sum(1 for _, _, _, t in matmul_info if t)
not_transposed_count = sum(1 for _, _, _, t in matmul_info if not t)
print(f'  Transposed: {transposed_count}, Not transposed: {not_transposed_count}')

# Show non-transposed ones (these might have wrong axes)
if not_transposed_count > 0:
    print('\nNon-transposed MatMul weights:')
    for name, wname, shape, t in matmul_info:
        if not t:
            print(f'  {wname}: {shape}')
