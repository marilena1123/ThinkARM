#!/bin/bash
#SBATCH --job-name=thinkarm_annotate
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=EUHPC_D33_235
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load cuda/12.2
module load anaconda3/2023.09-0

# Activate your environment
source activate /leonardo_work/EUHPC_D33_216/mzoumpou/cogchains_env

cd /home/user/ThinkARM

mkdir -p logs outputs_annotated

# Disable torch.compile to avoid dynamo bugs on HPC
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# -------------------------------------------------------
# Usage:
#   sbatch job_annotate.sh <input_file> [--thinking-only]
#
# Example:
#   sbatch job_annotate.sh outputs/gen_qwen3_30b_thinking_gsm8k.json
#   sbatch job_annotate.sh outputs/gen_deepseek_r1_distill_qwen_7b_math.json --thinking-only
# -------------------------------------------------------

INPUT_FILE="${1:-outputs/gen_qwen3_30b_thinking_gsm8k.json}"
THINKING_ONLY="${2:-}"

# Judge model path
JUDGE_MODEL="/leonardo_work/EUHPC_D33_215/step_saes/model/Llama-3.3-70B-Instruct"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

if [ ! -d "$JUDGE_MODEL" ]; then
    echo "ERROR: Judge model not found: $JUDGE_MODEL"
    exit 1
fi

echo "Annotating thinking traces: input=$INPUT_FILE"
echo "Judge model: $JUDGE_MODEL"

ARGS="--input_path $INPUT_FILE --judge_model_path $JUDGE_MODEL --output_dir outputs_annotated --batch_size 8"

if [ "$THINKING_ONLY" = "--thinking-only" ]; then
    ARGS="$ARGS --annotate_thinking_only"
    echo "Annotating thinking traces only"
fi

python annotate_thinking_traces.py \
    $ARGS \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.90 \
    --save_every 10
