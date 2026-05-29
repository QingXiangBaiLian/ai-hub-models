"""
Quantize Qwen3.5-0.8B with W8A16 (no SpinQuant) and evaluate PPL.
Combined script to avoid OOM from save step.
"""
import sys
import os
import gc
import traceback

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/workspace/ai-hub-models/src')

import math
import torch
from torch.nn import CrossEntropyLoss
from qai_hub_models.models.qwen3_5_0_8b import Model
from qai_hub_models.models.qwen3_5_0_8b.model import Qwen3_5_0_8B as FPModelClass
from qai_hub_models.models.common import Precision

SEQ_LEN = 128
CTX_LEN = 256

print("=" * 60, flush=True)
print("Quantizing Qwen3.5-0.8B with W8A16 (no SpinQuant) + PPL Eval", flush=True)
print("=" * 60, flush=True)

try:
    # Load FP model
    print("\n[1/5] Loading FP model...", flush=True)
    fp_model = FPModelClass.from_pretrained(
        sequence_length=SEQ_LEN,
        context_length=CTX_LEN,
        host_device=torch.device("cpu"),
    )

    # Save tokenizer and config before deleting fp_model
    tokenizer = fp_model.tokenizer
    llm_config = fp_model.llm_config

    # Create quantized model
    print("\n[2/5] Creating quantized model (W8A16)...", flush=True)
    model_quant = Model.from_pretrained(
        checkpoint=None,
        host_device=torch.device("cpu"),
        sequence_length=SEQ_LEN,
        context_length=CTX_LEN,
        precision=Precision.w8a16,
        fp_model=fp_model,
    )

    # Free fp_model ASAP
    print("Freeing FP model memory...", flush=True)
    del fp_model
    gc.collect()

    # Run calibration (no SpinQuant)
    print("\n[3/5] Running calibration...", flush=True)
    model_quant.quantize(use_spin_quant=False)
    print("Calibration done!", flush=True)
    gc.collect()

    # Run PPL evaluation directly (without LLM_Generator to avoid memory leak)
    print("\n[4/5] Setting up PPL evaluation...", flush=True)
    from qai_hub_models.models._shared.llm.generator import LLM_Generator
    from qai_hub_models.datasets import get_dataset_from_name
    from qai_hub_models.datasets.common import DatasetSplit
    from torch.utils.data import DataLoader

    # Create embedding
    EmbeddingClass = FPModelClass.EmbeddingClass
    embedding = EmbeddingClass(max_length=CTX_LEN, config=llm_config)

    # Create generator
    generator = LLM_Generator(
        [model_quant],
        tokenizer,
        embedding,
        accumulate_logits_on_cpu=True,
    )

    # Create dataset (10 samples to avoid memory leak)
    NUM_SAMPLES = 10
    dataset = get_dataset_from_name(
        name="wikitext",
        tokenizer=tokenizer,
        block_size=SEQ_LEN,
        context_length=CTX_LEN,
        num_samples=NUM_SAMPLES,
        split=DatasetSplit.TEST,
    )
    eval_dataloader = DataLoader(
        dataset, shuffle=False, batch_size=1, collate_fn=dataset.collate_fn
    )

    # Run evaluation manually with gc between samples
    print(f"\n[5/5] Running PPL evaluation ({NUM_SAMPLES} samples)...", flush=True)
    total_loss = 0.0
    num_batches = 0
    loss_fct = CrossEntropyLoss()

    for i, (input_ids, attention_mask, ground_truth) in enumerate(eval_dataloader):
        if i >= NUM_SAMPLES:
            break
        with torch.no_grad():
            outputs = generator(input_ids, attention_mask)
        logits = outputs.logits.reshape(1, -1, outputs.logits.shape[-1])
        shift_logits = logits[..., :-1, :].contiguous().to(dtype=torch.float32)
        shift_labels = ground_truth[..., 1:].contiguous()
        loss_value = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).item()
        total_loss += loss_value
        num_batches += 1
        print(f"  Sample {i+1}/{NUM_SAMPLES}: loss={loss_value:.4f}", flush=True)
        # Explicit cleanup to prevent memory leak
        del outputs, logits, shift_logits, shift_labels
        gc.collect()

    ppl = math.exp(total_loss / num_batches)
    print(f"\n{'=' * 60}", flush=True)
    print(f"RESULT: PPL (lower is better): {ppl:.2f}", flush=True)
    print(f"PPL = {ppl:.4f}", flush=True)
    print(f"{'=' * 60}", flush=True)

except Exception as e:
    print(f"\n\nERROR: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
