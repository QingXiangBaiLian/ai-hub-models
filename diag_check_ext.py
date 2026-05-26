import onnx
model = onnx.load('/workspace/qwen3_5_0_8b_w4a16_static/model_seqlen128_cl256.onnx', load_external_data=False)
ext_files = set()
for t in model.graph.initializer:
    for entry in t.external_data:
        if entry.key == 'location':
            ext_files.add(entry.value)
print('External data files referenced:', ext_files)
print('Num initializers:', len(model.graph.initializer))
