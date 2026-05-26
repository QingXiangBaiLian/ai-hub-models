import sys, os, math, torch, gc
sys.path.insert(0, '/workspace/ai-hub-models/src')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B_AIMETOnnx, Qwen3_5_0_8B, LAYER_TYPES
from qai_hub_models.models._shared.llm.model import get_onnx_model
import onnxruntime as ort
import onnx

# Create FP model
fp_model = Qwen3_5_0_8B.from_pretrained(sequence_length=128, context_length=256).to('cpu').eval()

# Get input spec
spec = fp_model.get_input_spec(
    llm_config=fp_model.llm_config.to_dict(),
    sequence_length=128,
    context_length=256,
)
spec_names = list(spec.keys())
print(f'Input spec order ({len(spec_names)} names):')
for i, name in enumerate(spec_names):
    print(f'  [{i:2d}] {name}: {spec[name][0]}')

# Export ONNX
import tempfile
tmp = tempfile.mkdtemp()
onnx_path = os.path.join(tmp, 'model.onnx')
onnx_model = get_onnx_model(fp_model, context_length=256, sequence_length=128, path=onnx_path, return_model=True, llm_io_type=fp_model.llm_io_type, use_dynamic_shapes=False)
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
session_names = [inp.name for inp in session.get_inputs()]
print(f'\nSession input order ({len(session_names)} names):')
for i, name in enumerate(session_names):
    shape = session.get_inputs()[i].shape
    print(f'  [{i:2d}] {name}: {shape}')

# Check for differences
print(f'\nOrdering matches: {spec_names == session_names}')
if spec_names != session_names:
    for i, (s, o) in enumerate(zip(spec_names, session_names)):
        if s != o:
            print(f'  MISMATCH at index {i}: spec={s} vs session={o}')
