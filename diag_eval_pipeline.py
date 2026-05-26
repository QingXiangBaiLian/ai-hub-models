"""Diagnostic script that exactly mimics the evaluation pipeline.
Processes first few wikitext samples and prints per-sample diagnostics.
"""
import torch
import math
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers.cache_utils import DynamicCache

# Step 1: Load FP model (same as evaluate())
print("=" * 60)
print("STEP 1: Load FP model")
print("=" * 60)
from qai_hub_models.models.qwen3_5_0_8b.model import (
    Qwen3_5_0_8B,
    Qwen3_5_0_8B_AIMETOnnx,
)

SEQ_LEN = 128
CTX_LEN = 256

fp_model = Qwen3_5_0_8B.from_pretrained(
    sequence_length=SEQ_LEN,
    context_length=CTX_LEN,
)
print(f"FP model loaded: seq_len={fp_model.sequence_length}, ctx_len={fp_model.context_length}")

# Step 2: Get dataset (same as evaluate())
print("\n" + "=" * 60)
print("STEP 2: Load wikitext dataset")
print("=" * 60)
from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets import get_dataset_from_name

dataset = get_dataset_from_name(
    name="wikitext",
    tokenizer=fp_model.tokenizer,
    block_size=fp_model.sequence_length,
    context_length=fp_model.context_length,
    num_samples=5,  # Just first 5 samples
    split=DatasetSplit.TEST,
)
eval_dataloader = DataLoader(
    dataset, shuffle=False, batch_size=1, collate_fn=dataset.collate_fn
)
print(f"Dataset loaded: {len(dataset)} samples")
print(f"First sample keys: {list(dataset[0].keys())}")
sample0 = dataset[0]
print(f"First sample input_ids shape: {sample0['input_ids'].shape}")
print(f"First sample attention_mask shape: {sample0['attention_mask'].shape}")
print(f"First 20 tokens: {sample0['input_ids'][0, :20].tolist()}")

# Step 3: Get evaluator (same as evaluate())
print("\n" + "=" * 60)
print("STEP 3: Create evaluator")
print("=" * 60)
evaluator = fp_model.get_evaluator("wikitext", torch.device("cpu"))
print(f"Evaluator type: {type(evaluator).__name__}")

# Step 4: Load quantized model (same as evaluate())
print("\n" + "=" * 60)
print("STEP 4: Load quantized model")
print("=" * 60)
del fp_model  # Free memory
import gc
gc.collect()

model = Qwen3_5_0_8B_AIMETOnnx.from_pretrained(
    host_device=torch.device("cpu"),
    sequence_length=SEQ_LEN,
    context_length=CTX_LEN,
    precision="w4a16",
    checkpoint="/workspace/qwen3_5_0_8b_w4a16_static",
)
model.eval()
print(f"Quantized model loaded: seq_len={model.sequence_length}, ctx_len={model.context_length}")
print(f"attention_mask_min_clip={model.attention_mask_min_clip}")
print(f"attention_mask_multiplier={model.attention_mask_multiplier}")
print(f"llm_io_type={model.llm_io_type}")

# Step 5: Create embedding and generator (same as evaluate())
print("\n" + "=" * 60)
print("STEP 5: Create embedding and generator")
print("=" * 60)
from qai_hub_models.models._shared.qwen3_5.model import Qwen3_5RopeEmbedding
from qai_hub_models.models._shared.llm.generator import LLM_Generator

embedding = Qwen3_5RopeEmbedding(
    max_length=CTX_LEN,
    config=model.llm_config,
)
print(f"Embedding created: max_length={CTX_LEN}")

generator = LLM_Generator(
    [model],
    model.tokenizer,
    embedding,
    accumulate_logits_on_cpu=True,
)
print(f"Generator created")
print(f"Generator llm_io_type: {generator.llm_io_type}")
print(f"Generator selected_model: {type(generator.selected_model).__name__}")

# Step 6: Process samples (same as evaluator.for_each_batch)
print("\n" + "=" * 60)
print("STEP 6: Process samples through evaluation pipeline")
print("=" * 60)

total_loss = 0.0
num_batches = 0

for i, sample in enumerate(eval_dataloader):
    input_ids, attention_mask, ground_truth = sample
    print(f"\n--- Sample {i} ---")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")
    print(f"  ground_truth shape: {ground_truth.shape}")
    print(f"  input_ids[:10]: {input_ids[0, :10].tolist()}")
    print(f"  attention_mask[:10]: {attention_mask[0, :10].tolist()}")
    print(f"  ground_truth[:10]: {ground_truth[0, :10].tolist()}")
    print(f"  input_ids == ground_truth: {torch.equal(input_ids, ground_truth)}")
    
    inputs = [input_ids, attention_mask]
    inputs = [inp.to(torch.device("cpu")) for inp in inputs]
    
    with torch.no_grad():
        outputs = generator(*inputs)
    
    logits = outputs.logits
    print(f"  logits shape: {logits.shape}")
    print(f"  logits dtype: {logits.dtype}")
    print(f"  logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}, std={logits.std():.4f}")
    print(f"  Any NaN: {torch.isnan(logits).any()}, Any Inf: {torch.isinf(logits).any()}")
    
    # Check a few specific logit positions
    print(f"  logits[0, 0, :5]: {logits[0, 0, :5].tolist()}")  # First position
    print(f"  logits[0, 127, :5]: {logits[0, 127, :5].tolist()}")  # Last of first chunk
    if logits.shape[1] > 128:
        print(f"  logits[0, 128, :5]: {logits[0, 128, :5].tolist()}")  # First of second chunk
        print(f"  logits[0, -1, :5]: {logits[0, -1, :5].tolist()}")  # Last position
    
    # Compute loss the same way as PerplexityEvaluator.add_batch
    lm_logits = logits.reshape(1, -1, logits.shape[-1])
    shift_logits = lm_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
    shift_labels = ground_truth[..., 1:].contiguous().to(shift_logits.device)
    
    print(f"  shift_logits shape: {shift_logits.shape}")
    print(f"  shift_labels shape: {shift_labels.shape}")
    print(f"  shift_labels[:10]: {shift_labels[0, :10].tolist()}")
    
    loss_fct = CrossEntropyLoss()
    loss_value = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).item()
    
    sample_ppl = math.exp(loss_value)
    print(f"  loss: {loss_value:.4f}")
    print(f"  per-sample PPL: {sample_ppl:.2f}")
    
    # Also check the first chunk alone
    first_chunk_logits = lm_logits[:, :128, :]
    first_chunk_shift_logits = first_chunk_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
    first_chunk_shift_labels = ground_truth[..., 1:128].contiguous()
    loss_chunk1 = CrossEntropyLoss()(
        first_chunk_shift_logits.view(-1, first_chunk_shift_logits.size(-1)),
        first_chunk_shift_labels.view(-1),
    ).item()
    print(f"  first chunk (0-127) loss: {loss_chunk1:.4f}, PPL: {math.exp(loss_chunk1):.2f}")
    
    if logits.shape[1] > 128:
        second_chunk_logits = lm_logits[:, 128:, :]
        second_chunk_shift_logits = second_chunk_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
        second_chunk_shift_labels = ground_truth[..., 129:].contiguous()
        loss_chunk2 = CrossEntropyLoss()(
            second_chunk_shift_logits.view(-1, second_chunk_shift_logits.size(-1)),
            second_chunk_shift_labels.view(-1),
        ).item()
        print(f"  second chunk (128-255) loss: {loss_chunk2:.4f}, PPL: {math.exp(loss_chunk2):.2f}")
    
    total_loss += loss_value
    num_batches += 1
    
    if i >= 2:  # Only process first 3 samples
        break

average_loss = total_loss / num_batches
overall_ppl = math.exp(average_loss)
print(f"\n{'=' * 60}")
print(f"OVERALL: average_loss={average_loss:.4f}, PPL={overall_ppl:.2f}")
print(f"{'=' * 60}")
