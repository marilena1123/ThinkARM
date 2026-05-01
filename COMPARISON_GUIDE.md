# ThinkARM vs. BLOOM Comparison Workflow

Complete guide for generating thinking traces and comparing two annotation frameworks:
- **ThinkARM**: Episode-based reasoning analysis (Schoenfeld's Episode Theory)
- **BLOOM**: Cognitive taxonomy-based reasoning analysis (Bloom's Taxonomy)

## Quick Overview

```
Thinking Traces (from models)
         ↓
    [generate.py]
         ↓
Raw traces with thinking_steps
         ↓
    [convert_for_bloom.py]  →  Reconstruct full thinking text
         ↓
Formatted for both frameworks
         ↓
    [annotate_thinkarm.py]     [annotate_bloom.py]
         ↓                              ↓
ThinkARM Episodes              BLOOM Cognitive Levels
(Read, Analyze, Explore...)    (Remember, Understand, Apply...)
         ↓                              ↓
    thinkarm_labels            bloom_labels
    thinking_thinkarm_labels   thinking_bloom_labels
```

## Step 1: Generate Thinking Traces

Generate raw thinking traces from your reasoning models:

```bash
# Quick test (small model, small dataset)
python generate.py \
    --model_path /leonardo_work/EUHPC_D33_216/mzoumpou/model/Qwen3-4b-thinking \
    --model_name qwen3_4b_thinking \
    --dataset_path /leonardo_work/EUHPC_D33_216/mzoumpou/datasets/gsm8k.json \
    --dataset_name gsm8k \
    --output_dir outputs \
    --batch_size 16 \
    --tensor_parallel_size 1

# Or on cluster
sbatch job_generate.sh qwen3_30b_thinking gsm8k
```

**Output**: `outputs/gen_qwen3_30b_thinking_gsm8k.json`

Each result contains:
```json
{
  "question": "...",
  "reasoning": "...",
  "thinking_trace": "...",
  "thinking_steps": ["step1", "step2", ...],
  "steps": ["final step1", "final step2", ...],
  "correct": true,
  ...
}
```

## Step 2: Convert for BLOOM Annotation

Reconstruct full thinking text from steps for BLOOM annotation:

```bash
python convert_for_bloom.py \
    --input outputs/gen_qwen3_30b_thinking_gsm8k.json \
    --output outputs/bloom_input_qwen3_30b_thinking_gsm8k.json
```

**Why this step?**
- BLOOM annotator expects full text, not sentence arrays
- Need to reconstruct `thinking_trace` from `thinking_steps`
- Prepare `reasoning` from `steps`
- Preserve all other fields

**Output**: `outputs/bloom_input_qwen3_30b_thinking_gsm8k.json`

Same structure, but with reconstructed full texts:
```json
{
  "question": "...",
  "reasoning": "full reconstructed text from steps",
  "thinking_trace": "full reconstructed text from thinking_steps",
  "correct": true,
  ...
}
```

## Step 3: Annotate with Both Frameworks

### Option A: ThinkARM Annotation (Direct)

```bash
# Annotate with ThinkARM episodes
python annotate_thinking_traces.py \
    --input_path outputs/gen_qwen3_30b_thinking_gsm8k.json \
    --judge_model_path /leonardo_work/EUHPC_D33_215/step_saes/model/Llama-3.3-70B-Instruct \
    --output_dir outputs_annotated

# Or on cluster
sbatch job_annotate.sh outputs/gen_qwen3_30b_thinking_gsm8k.json
```

**Output**: `outputs_annotated/gen_qwen3_30b_thinking_gsm8k_annotated.json`

Adds fields:
```json
{
  "thinkarm_labels": [
    {"index": 1, "category": "Read", "reason": "..."},
    {"index": 2, "category": "Analyze", "reason": "..."},
    ...
  ],
  "thinking_thinkarm_labels": [
    {"index": 1, "category": "Explore", "reason": "..."},
    ...
  ],
  "thinkarm_judge": "llama3_70b_instruct",
  "thinkarm_judge_version": "v1_thinkarm_vllm",
  ...
}
```

### Option B: BLOOM Annotation

```bash
# Annotate with Bloom's taxonomy
python annotate_bloom.py \
    --judge_model_path /leonardo_work/EUHPC_D33_215/step_saes/model/Llama-3.3-70B-Instruct \
    --input_dir outputs \
    --pattern "bloom_input_*.json" \
    --output_dir outputs_bloom_annotated

# Or on cluster
sbatch job_annotate_bloom.sh outputs "bloom_input_*.json"
```

**Output**: `outputs_bloom_annotated/bloom_input_qwen3_30b_thinking_gsm8k.json`

Adds fields:
```json
{
  "bloom_labels": [
    {"step": "full text", "bloom_level": "ANALYZE", "bloom_reasoning": "..."},
    {"step": "full text", "bloom_level": "APPLY", "bloom_reasoning": "..."},
    ...
  ],
  "thinking_bloom_labels": [
    {"step": "full text", "bloom_level": "EVALUATE", "bloom_reasoning": "..."},
    ...
  ],
  "bloom_judge": "llama3_70b",
  "bloom_judge_version": "v2_bloom_only_vllm_matched",
  ...
}
```

## Step 4: Compare the Annotations

### Framework Mapping

| ThinkARM Episode | BLOOM Level | Purpose |
|------------------|------------|---------|
| Read | REMEMBER | Understanding the problem |
| Analyze | ANALYZE | Breaking down problem structure |
| Explore | UNDERSTAND | Exploring alternatives |
| Plan | UNDERSTAND | Planning approach |
| Implement | APPLY | Executing procedures |
| Verify | EVALUATE | Checking correctness |
| Monitor | EVALUATE | Monitoring progress |
| Answer | REMEMBER | Stating solution |

### Analysis Example

```python
import json
from collections import Counter

# Load both annotated files
with open("outputs_annotated/gen_qwen3_30b_thinking_gsm8k_annotated.json") as f:
    thinkarm = json.load(f)

with open("outputs_bloom_annotated/bloom_input_qwen3_30b_thinking_gsm8k.json") as f:
    bloom = json.load(f)

# Compare first result
result_ta = thinkarm["results"][0]
result_bl = bloom["results"][0]

print("ThinkARM Episode Distribution:")
episodes = Counter(l["category"] for l in result_ta["thinkarm_labels"])
for ep, count in episodes.most_common():
    print(f"  {ep}: {count}")

print("\nBLOOM Level Distribution:")
blooms = Counter(l["bloom_level"] for l in result_bl["bloom_labels"])
for bl, count in blooms.most_common():
    print(f"  {bl}: {count}")

# Compare reasoning labels
print("\nFirst 3 reasoning steps comparison:")
for i in range(3):
    ta_label = result_ta["thinkarm_labels"][i]
    bl_label = result_bl["bloom_labels"][i]
    
    print(f"\nStep {i+1}:")
    print(f"  Text: {ta_label['sentence'][:50]}...")
    print(f"  ThinkARM: {ta_label['category']}")
    print(f"  BLOOM: {bl_label['bloom_level']}")
```

## Multi-GPU Cluster Submission

Submit multiple annotation jobs for comprehensive comparison:

```bash
# Generate traces from multiple models
sbatch job_generate.sh qwen3_30b_thinking gsm8k
sbatch job_generate.sh qwen3_30b_thinking math
sbatch job_generate.sh deepseek_r1_distill_qwen_7b gsm8k

# Wait for generation to complete, then convert all
for f in outputs/gen_*.json; do
    echo "Converting $f..."
    python convert_for_bloom.py --input "$f" --output "outputs/bloom_$(basename $f)"
done

# Annotate with both frameworks
sbatch job_annotate.sh outputs/gen_qwen3_30b_thinking_gsm8k.json
sbatch job_annotate_bloom.sh outputs "bloom_input_*.json"

# Then compare
python compare_frameworks.py \
    --thinkarm outputs_annotated \
    --bloom outputs_bloom_annotated
```

## Understanding the Difference

### ThinkARM (Episode-based)
- **Focus**: Functional reasoning steps in mathematical problem-solving
- **Categories**: 8 episodes (Read, Analyze, Explore, Plan, Implement, Verify, Monitor, Answer)
- **Model**: Based on Schoenfeld's Episode Theory
- **Grain**: Sentence-level segmentation
- **Output**: `sentence-category` with `sentence-category-reason`

### BLOOM (Taxonomy-based)
- **Focus**: General cognitive processing levels
- **Categories**: 6 levels (Remember, Understand, Apply, Analyze, Evaluate, Create)
- **Model**: Bloom's revised taxonomy
- **Grain**: Cognitive function-based segmentation (can span multiple sentences)
- **Output**: `bloom_level` with `bloom_reasoning`

### When to use each:

**ThinkARM is better for:**
- Analyzing mathematical reasoning specifically
- Understanding problem-solving strategies
- Comparing thinking models vs. non-thinking models
- Episode-level temporal dynamics

**BLOOM is better for:**
- General cognitive load analysis
- Comparing with educational assessments
- Understanding knowledge level progression
- Cross-domain reasoning analysis

## Output Comparison Structure

After both annotations, your result looks like:

```json
{
  "question": "Solve: 2x + 3 = 7",
  "reasoning": "Let me solve for x. Start with 2x + 3 = 7. Subtract 3: 2x = 4. Divide by 2: x = 2.",
  "thinking_trace": "First I need to isolate x. Let me subtract 3 from both sides...",
  "correct": true,
  
  // ThinkARM annotations (sentence-level episodes)
  "thinkarm_labels": [
    {"index": 1, "sentence": "Let me solve for x.", "category": "Read", "reason": "..."},
    {"index": 2, "sentence": "Start with 2x + 3 = 7. Subtract 3: 2x = 4.", "category": "Implement", "reason": "..."},
    {"index": 3, "sentence": "Divide by 2: x = 2.", "category": "Implement", "reason": "..."}
  ],
  
  // BLOOM annotations (cognitive level-based segmentation)
  "bloom_labels": [
    {"step": "Let me solve for x.", "bloom_level": "ANALYZE", "bloom_reasoning": "..."},
    {"step": "Start with 2x + 3 = 7. Subtract 3: 2x = 4. Divide by 2: x = 2.", "bloom_level": "APPLY", "bloom_reasoning": "..."}
  ],
  
  // Same for thinking traces
  "thinking_thinkarm_labels": [...],
  "thinking_bloom_labels": [...]
}
```

## Troubleshooting

### Convert script issues
- Check that thinking_steps exist in input
- Verify reasoning/steps fields are present

### BLOOM annotation fails
- Ensure text is plain (no special tokens)
- Check that judge_model_path is correct
- Verify patterns match file names

### Comparison mismatches
- ThinkARM segments by sentence boundaries
- BLOOM segments by cognitive shifts (can span sentences)
- Both use different granularities by design

## Next Steps

1. **Merge results**: Combine multiple files for cross-model analysis
2. **Statistical comparison**: Compare episode vs. BLOOM distributions
3. **Correctness analysis**: Correlate annotation patterns with answer correctness
4. **Visualization**: Plot temporal dynamics of episodes vs. bloom levels
5. **Publication**: Analyze differences for research paper
