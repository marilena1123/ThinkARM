# ThinkARM Thinking Trace Generation & Annotation Pipeline

Complete pipeline for generating thinking traces with reasoning models and annotating them with ThinkARM episode labels.

## Overview

This pipeline has three main stages:

1. **Generation**: Generate thinking traces from reasoning models using vLLM
2. **Annotation**: Annotate the traces with ThinkARM episode categories using a judge model
3. **Analysis**: Compute statistics and analyze the annotated results

## Prerequisites

- Local HPC cluster with GPU access (tested on Leonardo)
- Local model files in HuggingFace format
- Local dataset files in JSON format
- vLLM and other dependencies installed

## Directory Structure

```
ThinkARM/
├── generate.py                 # Generate thinking traces
├── annotate_thinking_traces.py # Annotate traces with episodes
├── data_utils.py              # Analysis and data utilities
├── job_generate.sh            # SLURM script for generation
├── job_annotate.sh            # SLURM script for annotation
├── outputs/                    # Generated traces
├── outputs_annotated/          # Annotated traces
└── logs/                       # SLURM logs
```

## Step 1: Generate Thinking Traces

### Quick Start

```bash
# Run locally (small test)
python generate.py \
    --model_path /path/to/model \
    --model_name qwen3_30b_thinking \
    --dataset_path /path/to/gsm8k.json \
    --dataset_name gsm8k \
    --output_dir outputs

# Or submit to cluster
sbatch job_generate.sh qwen3_30b_thinking gsm8k
```

### Options

```
--model_path PATH              Path to locally saved HF model (required)
--model_name NAME              Short model name for filenames (required)
--dataset_path PATH            Path to dataset JSON (required)
--dataset_name NAME            Dataset name (e.g., gsm8k, math) (required)
--output_dir DIR               Output directory (default: outputs)
--max_new_tokens N             Max tokens to generate (default: 2048)
--temperature T                Generation temperature (default: 0.7)
--batch_size N                 Batch size (default: 32)
--tensor_parallel_size N       GPUs for parallelism (default: 1)
--gpu_memory_utilization F     GPU memory fraction (default: 0.90)
--save_every N                 Save checkpoint every N samples (default: 50)
--start_idx N                  Resume from index N
--end_idx N                    Process up to index N
```

### Supported Models

- `qwen3_30b_thinking`: Qwen3 30B with thinking
- `qwen3_4b_thinking`: Qwen3 4B with thinking
- `phi_4_reasoning`: Phi-4 Reasoning model
- `deepseek_r1_distill_qwen_1.5b`: DeepSeek R1 Distill
- `deepseek_r1_distill_qwen_7b`: DeepSeek R1 Distill
- `deepseek_r1_distill_llama8b`: DeepSeek R1 Distill

### Supported Datasets

- `gsm8k`: GSM8K math dataset
- `gsm_hard`: Hard GSM8K variants
- `math`: MATH dataset

### Output Format

```json
{
  "summary": {
    "_summary": true,
    "model": "qwen3_30b_thinking",
    "dataset": "gsm8k",
    "total": 100,
    "correct": 75,
    "accuracy": 0.75
  },
  "results": [
    {
      "question": "What is 2+2?",
      "reasoning": "Let me calculate... 2+2=4",
      "predicted_answer": "4",
      "ground_truth": "4",
      "correct": true,
      "task": "arithmetic",
      "thinking_trace": "Let me think... 2 and 2 makes 4",
      "thinking_steps": ["Let me think...", "2 and 2 makes 4"],
      "thinking_predicted_answer": "4",
      "thinking_correct": true,
      "steps": ["Let me calculate...", "2+2=4"]
    }
  ]
}
```

## Step 2: Annotate with ThinkARM

ThinkARM uses 8 episode categories to analyze reasoning:

- **Read**: Understanding the problem statement
- **Analyze**: Analyzing problem structure and key elements
- **Explore**: Exploring different approaches and strategies
- **Plan**: Planning the solution approach
- **Implement**: Concrete execution and calculations
- **Verify**: Verifying correctness
- **Monitor**: Monitoring progress and evaluating intermediate results
- **Answer**: Providing the final answer

### Quick Start

```bash
# Run locally
python annotate_thinking_traces.py \
    --input_path outputs/gen_qwen3_30b_thinking_gsm8k.json \
    --judge_model_path /path/to/Llama-3.3-70B-Instruct \
    --output_dir outputs_annotated

# Or submit to cluster
sbatch job_annotate.sh outputs/gen_qwen3_30b_thinking_gsm8k.json
```

### Options

```
--input_path PATH              Path to generated traces (required)
--judge_model_path PATH        Path to judge model (required)
--output_dir DIR               Output directory (default: outputs_annotated)
--batch_size N                 Annotation batch size (default: 8)
--tensor_parallel_size N       GPUs for parallelism (default: 1)
--gpu_memory_utilization F     GPU memory fraction (default: 0.90)
--annotate_thinking_only       Only annotate thinking traces
--save_every N                 Save checkpoint every N (default: 10)
--start_idx N                  Resume from index N
```

### Output Format

The annotated results include the original fields plus:

```json
{
  "question": "...",
  "reasoning": "...",
  "correct": true,
  "thinking_trace": "...",
  "thinkarm_labels": [
    {
      "index": 1,
      "category": "Read",
      "reason": "The solver identifies the problem statement"
    },
    {
      "index": 2,
      "category": "Analyze",
      "reason": "The solver breaks down the problem structure"
    },
    {
      "index": 3,
      "category": "Implement",
      "reason": "The solver performs calculations"
    }
  ],
  "thinking_thinkarm_labels": [
    {
      "index": 1,
      "category": "Explore",
      "reason": "The solver explores different approaches"
    }
  ],
  "thinkarm_judge": "llama3_70b_instruct",
  "thinkarm_judge_version": "v1_thinkarm_vllm",
  "thinkarm_judge_raw": "..."
}
```

## Step 3: Analysis

Use `data_utils.py` to analyze the results.

### Print Statistics

```bash
python data_utils.py stats --path outputs_annotated/gen_qwen3_30b_thinking_gsm8k_annotated.json
```

Output:
```
============================================================
GENERATION STATISTICS
============================================================
Total results: 1000
Correct: 750
Accuracy: 0.7500

Episode Distribution (Final Reasoning):
  Implement: 15000 (40.5%)
  Verify: 8500 (23.0%)
  Monitor: 6200 (16.8%)
  Analyze: 4100 (11.1%)
  Answer: 2200 (5.9%)
  Explore: 1800 (4.9%)
  Plan: 1500 (4.1%)
  Read: 900 (2.4%)

Episode Distribution (Thinking Traces):
  Explore: 5200 (35.1%)
  Implement: 4800 (32.4%)
  Monitor: 2100 (14.2%)
  Verify: 1900 (12.8%)
  ...
```

### Merge Results

```bash
python data_utils.py merge \
    --inputs outputs_annotated/gen_qwen3_30b_thinking_gsm8k_annotated.json \
               outputs_annotated/gen_deepseek_r1_distill_qwen_7b_math_annotated.json \
    --output outputs_annotated/merged_all.json
```

### Filter Results

```bash
# Keep only correct answers
python data_utils.py filter \
    --path outputs_annotated/gen_qwen3_30b_thinking_gsm8k_annotated.json \
    --output outputs_annotated/correct_only.json \
    --correct

# Keep only a specific task
python data_utils.py filter \
    --path outputs_annotated/merged_all.json \
    --output outputs_annotated/formal_fallacies_only.json \
    --task formal_fallacies
```

## SLURM Submission Examples

### Generate on cluster

```bash
# Single model/dataset pair
sbatch job_generate.sh qwen3_30b_thinking gsm8k

# Phi-4 needs different tensor parallelism
sbatch job_generate.sh phi_4_reasoning math

# Multiple submissions for full evaluation
sbatch job_generate.sh qwen3_30b_thinking gsm8k
sbatch job_generate.sh qwen3_30b_thinking math
sbatch job_generate.sh qwen3_30b_thinking gsm_hard
sbatch job_generate.sh deepseek_r1_distill_qwen_7b gsm8k
sbatch job_generate.sh deepseek_r1_distill_qwen_7b math
sbatch job_generate.sh phi_4_reasoning gsm8k
```

### Annotate on cluster

```bash
# Wait for generation to complete first
sbatch job_annotate.sh outputs/gen_qwen3_30b_thinking_gsm8k.json

# Annotate only thinking traces (faster)
sbatch job_annotate.sh outputs/gen_qwen3_30b_thinking_math.json --thinking-only
```

## Resume from Checkpoint

All scripts support resuming from checkpoints:

### Generation

```bash
# Resume from sample 500
python generate.py \
    --model_path /path/to/model \
    --model_name qwen3_30b_thinking \
    --dataset_path /path/to/gsm8k.json \
    --dataset_name gsm8k \
    --output_dir outputs \
    --start_idx 500
```

### Annotation

```bash
# Resume annotation from sample 200
python annotate_thinking_traces.py \
    --input_path outputs/gen_qwen3_30b_thinking_gsm8k.json \
    --judge_model_path /path/to/judge \
    --output_dir outputs_annotated \
    --start_idx 200
```

## Performance Tips

1. **Batch Size**: Increase batch size for generation (32-64) for better GPU utilization
2. **Tensor Parallelism**: Use 4 GPUs for large models (70B+)
3. **Checkpoints**: Generate saves checkpoints every 50 samples; annotation every 10
4. **Temperature**: Use lower temperature (0.3) for judge model to get consistent annotations
5. **GPU Memory**: Adjust `--gpu_memory_utilization` based on your GPU (0.80-0.95)

## Troubleshooting

### vLLM CUDA errors with MoE models

```bash
export VLLM_USE_V1=0
python generate.py ...
```

### Out of memory

- Reduce batch size
- Reduce max_new_tokens
- Reduce gpu_memory_utilization
- Use tensor parallelism

### Slow annotation

- Increase batch size (but watch memory)
- Use thinking-only mode
- Use a smaller judge model

## Dataset Format

Expected JSON format for datasets:

```json
[
  {
    "question": "What is 2+2?",
    "answer": "4"
  },
  {
    "question": "Solve: x + 5 = 10",
    "answer": "5",
    "task": "algebra"
  }
]
```

## Citation

If you use this pipeline, please cite:

```bibtex
@article{schoenfeld2024thinkarm,
  title={Schoenfeld's Anatomy of Mathematical Reasoning by Language Models},
  author={...},
  year={2024}
}
```

## Support

For issues or questions, check:
- SLURM logs in `logs/` directory
- Generated checkpoints for error context
- `--start_idx` to resume from specific point
