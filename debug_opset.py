"""Debug: check opset in the exported ONNX model."""
import sys
sys.path.insert(0, "/workspace/ai-hub-models/src")

import onnx
import torch
import numpy as np

from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B

print("Loading model...")
fp_model = Qwen3_5_0_8B.from_pretrained()

# Minimal export to check opset
from qai_hub_models.models._shared.llm.model import get_onnx_model
import tempfile, os

tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "model.onnx")

print("Exporting ONNX (optimize=False)...")
onnx_model = get_onnx_model(
    fp_model=fp_model,
    context_length=4096,
    sequence_length=2048,
    path=path,
    return_model=True,
    llm_io_type=fp_model.llm_io_type,
    use_dynamic_shapes=True,
)

print(f"\nModel opset_import: {list(onnx_model.opset_import)}")
print(f"Number of opset entries: {len(onnx_model.opset_import)}")
for i, op in enumerate(onnx_model.opset_import):
    print(f"  [{i}] domain='{op.domain}', version={op.version}")

print(f"\nModel ir_version: {onnx_model.ir_version}")
print(f"Model graph name: {onnx_model.graph.name}")
print(f"Number of nodes: {len(onnx_model.graph.node)}")
print(f"Number of inputs: {len(onnx_model.graph.input)}")

# Try to create ort session
print("\nTrying to create ORT session...")
import onnxruntime as ort
try:
    # Save to temp file with external data
    tmppath = os.path.join(tmpdir, "test_model.onnx")
    onnx.save_model(onnx_model, tmppath, save_as_external_data=True, all_tensors_to_one_file=True, location="test_model.data")
    sess = ort.InferenceSession(tmppath, providers=["CPUExecutionProvider"])
    print("SUCCESS: ORT session created!")
    print(f"  Inputs: {[i.name for i in sess.get_inputs()]}")
except Exception as e:
    print(f"FAILED: {e}")

# cleanup
import shutil
shutil.rmtree(tmpdir)
