# Quick Start Guide

## 1. Generate Thinking Traces (5-10 min setup)

```bash
# Simple test run (CPU-friendly, single GPU)
python generate.py \
    --model_path /leonardo_work/EUHPC_D33_216/mzoumpou/model/Qwen3-4b-thinking \
    --model_name qwen3_4b_thinking \
    --dataset_path /leonardo_work/EUHPC_D33_216/mzoumpou/datasets/gsm8k.json \
    --dataset_name gsm8k \
    --output_dir outputs \
    --batch_size 16 \
    --tensor_parallel_size 1

# Submit to cluster (recommended)
sbatch job_generate.sh qwen3_30b_thinking gsm8k
```

## 2. Annotate with ThinkARM (2-5 min setup)

Wait for generation to complete, then:

```bash
# Annotate the generated traces
sbatch job_annotate.sh outputs/gen_qwen3_30b_thinking_gsm8k.json

# Or faster (thinking only):
sbatch job_annotate.sh outputs/gen_qwen3_30b_thinking_gsm8k.json --thinking-only
```

## 3. Analyze Results (1 min)

```bash
# Print statistics
python data_utils.py stats --path outputs_annotated/gen_qwen3_30b_thinking_gsm8k_annotated.json

# Merge results from multiple models
python data_utils.py merge \
    --inputs outputs_annotated/gen_qwen3_30b_thinking_gsm8k_annotated.json \
             outputs_annotated/gen_deepseek_r1_distill_qwen_7b_math_annotated.json \
    --output outputs_annotated/merged.json

# Filter results
python data_utils.py filter \
    --path outputs_annotated/merged.json \
    --output outputs_annotated/correct_only.json \
    --correct
```

## Model Recommendations

### For quick testing:
- `qwen3_4b_thinking` + `gsm8k` (fastest)
- 30-60 min on 1 GPU

### For quality results:
- `qwen3_30b_thinking` + `gsm8k` (best balance)
- 2-4 hours on 4 GPUs

### For comprehensive comparison:
- All models on all datasets
- ~48 hours total on cluster

## Output Files

After generation:
```
outputs/gen_qwen3_30b_thinking_gsm8k.json
├── summary: accuracy stats
└── results: 1000 samples with thinking traces
```

After annotation:
```
outputs_annotated/gen_qwen3_30b_thinking_gsm8k_annotated.json
├── All previous fields
├── thinkarm_labels: episode annotations for final reasoning
├── thinking_thinkarm_labels: episode annotations for thinking
└── thinkarm_judge: llama3_70b_instruct
```

## Common Commands

### Check job status
```bash
squeue -u $USER
sinfo -R
```

### View logs
```bash
tail -f logs/thinkarm_gen_*.out
tail -f logs/thinkarm_annotate_*.out
```

### Resume interrupted job
```bash
# Check how many completed
python data_utils.py stats --path outputs/gen_qwen3_30b_thinking_gsm8k.json

# Resume from that point
sbatch job_generate.sh qwen3_30b_thinking gsm8k  # Auto-resumes
```

## What the Annotations Mean

```json
{
  "thinkarm_labels": [
    {
      "index": 1,
      "category": "Read",
      "reason": "Solver identifies the problem"
    },
    {
      "index": 2,
      "category": "Implement",
      "reason": "Solver performs calculations"
    },
    {
      "index": 3,
      "category": "Verify",
      "reason": "Solver checks the answer"
    }
  ]
}
```

The categories represent different functional steps in the reasoning process, allowing you to analyze how models structure their thinking.

## Next Steps

1. See `PIPELINE_README.md` for detailed documentation
2. Check `data_utils.py` for analysis functions
3. Customize prompts in `generate.py` and `annotate_thinking_traces.py`
