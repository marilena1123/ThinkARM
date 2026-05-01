# Annotating ThinkARM Data with BLOOM

Quick workflow to get BLOOM annotations on the already-annotated ThinkARM data from the original repository.

## Overview

```
ThinkARM annotated sentences (per problem)
         ↓
[convert_thinkarm_to_bloom.py]
  Reconstruct full trace from sentences
  (Read + Analyze + Explore + ... = full text)
         ↓
BLOOM-formatted input
         ↓
[annotate_bloom.py]
  Judge segments and classifies
  into Bloom cognitive levels
         ↓
BLOOM annotations on same traces
```

## Step 1: Get ThinkARM Original Data

The original repository has annotated data at:
- `/data/label/` - Annotated sentences per model/problem
- `/data/raw/` - Raw model outputs (questions, answers)

If you don't have it locally:

```bash
# Clone the original repository
git clone https://github.com/MingLiiii/ThinkARM.git original_thinkarm

# Or if you already have it, just note the paths:
THINKARM_LABEL_DIR="/path/to/ThinkARM/data/label"
RAW_DIR="/path/to/ThinkARM/data/raw"
```

## Step 2: Convert ThinkARM to BLOOM Format

Reconstruct full traces from the annotated sentences:

```bash
python convert_thinkarm_to_bloom.py \
    --thinkarm_dir /path/to/ThinkARM/data/label \
    --raw_dir /path/to/ThinkARM/data/raw \
    --output_dir outputs_bloom_from_thinkarm \
    --models QwQ32B,deepseekR1,DSQwen32B \
    --dataset_name math
```

**What this does:**
1. Reads sentence-by-sentence ThinkARM annotations
2. Joins all sentences back into full reasoning traces
3. Preserves original ThinkARM annotations for comparison
4. Outputs in BLOOM annotator format

**Output:** `outputs_bloom_from_thinkarm/bloom_input_QwQ32B_math.json`

Each result contains:
```json
{
  "question": "Find all functions f(x)...",
  "reasoning": "Okay, so I have this problem... [full text reconstructed from sentences]",
  "thinking_trace": "[thinking sentences only]",
  "answer_trace": "[answer sentences only]",
  "ground_truth": "...",
  "thinkarm_original_sentences": [
    {"sentence": "...", "sentence-category": "Read", ...},
    {"sentence": "...", "sentence-category": "Analyze", ...},
    ...
  ]
}
```

### Optional: Limit to specific models

```bash
# Only convert QwQ32B
python convert_thinkarm_to_bloom.py \
    --thinkarm_dir ... \
    --raw_dir ... \
    --models QwQ32B

# Convert first 10 problems only (for testing)
python convert_thinkarm_to_bloom.py \
    --thinkarm_dir ... \
    --raw_dir ... \
    --max_samples 10
```

## Step 3: Annotate with BLOOM

Feed the reconstructed traces to BLOOM:

```bash
# Local run
python annotate_bloom.py \
    --judge_model_path /leonardo_work/EUHPC_D33_215/step_saes/model/Llama-3.3-70B-Instruct \
    --input_dir outputs_bloom_from_thinkarm \
    --pattern "bloom_input_*.json" \
    --output_dir outputs_bloom_annotated

# Or on cluster
sbatch job_annotate_bloom.sh outputs_bloom_from_thinkarm "bloom_input_*.json"
```

## Step 4: Compare Annotations

Now you have both annotations in the same file:

```json
{
  "question": "...",
  "reasoning": "...",
  
  // Original ThinkARM (sentence-level)
  "thinkarm_original_sentences": [
    {"sentence": "...", "sentence-category": "Read"},
    {"sentence": "...", "sentence-category": "Implement"},
    {"sentence": "...", "sentence-category": "Verify"}
  ],
  
  // BLOOM (cognitive level-based segmentation)
  "bloom_labels": [
    {"step": "...", "bloom_level": "ANALYZE"},
    {"step": "...", "bloom_level": "APPLY"},
    {"step": "...", "bloom_level": "EVALUATE"}
  ]
}
```

## Quick Analysis

```python
import json

# Load annotated data
with open("outputs_bloom_annotated/bloom_input_QwQ32B_math.json") as f:
    data = json.load(f)

result = data["results"][0]

print("Original ThinkARM annotation:")
for sent in result["thinkarm_original_sentences"]:
    print(f"  {sent['sentence'][:40]}...")
    print(f"    ThinkARM: {sent['sentence-category']}")

print("\nBLOOM annotation:")
for label in result["bloom_labels"]:
    print(f"  {label['step'][:40]}...")
    print(f"    BLOOM: {label['bloom_level']}")
```

## Full Example Workflow

```bash
# 1. Convert all available models
python convert_thinkarm_to_bloom.py \
    --thinkarm_dir original_thinkarm/data/label \
    --raw_dir original_thinkarm/data/raw \
    --output_dir outputs_bloom_from_thinkarm \
    --dataset_name math

# 2. Annotate with BLOOM
sbatch job_annotate_bloom.sh outputs_bloom_from_thinkarm "bloom_input_*.json"

# 3. Compare results
# Now each result has:
#   - thinkarm_original_sentences (with sentence-category)
#   - bloom_labels (with bloom_level)
#   - bloom_judge info
```

## What Gets Compared

### ThinkARM (from original data)
- **Granularity**: Sentence-level
- **Categories**: Read, Analyze, Explore, Plan, Implement, Verify, Monitor, Answer
- **Source**: Human annotated + auto-annotated with GPT models
- **Field**: `thinkarm_original_sentences`

### BLOOM (new annotations)
- **Granularity**: Cognitive function-level (can span multiple sentences)
- **Categories**: Remember, Understand, Apply, Analyze, Evaluate, Create
- **Source**: Llama-3.3-70B-Instruct judge
- **Fields**: `bloom_labels`, `thinking_bloom_labels`, `bloom_judge_raw`

## Troubleshooting

**Missing models?**
```bash
# List available models
ls original_thinkarm/data/label/
```

**No results?**
- Check that raw_dir contains matching JSON files
- Verify problem_id matching between label and raw directories
- Use `--max_samples 5` to test with small subset

**BLOOM annotation fails?**
- Ensure judge model path is correct
- Check that text is plain (no LaTeX or special tokens)
- Try smaller batch_size in annotate_bloom.py

## Next Steps

1. **Statistical comparison**: Correlate ThinkARM categories with BLOOM levels
2. **Temporal analysis**: Compare episode progression with cognitive level progression
3. **Model differences**: Analyze how different reasoning models differ across both frameworks
4. **Validation**: Check if both frameworks identify the same critical reasoning steps
