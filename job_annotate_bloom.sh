#!/bin/bash
#SBATCH --job-name=thinkarm_bloom
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

mkdir -p logs outputs_bloom_annotated

# Disable torch.compile to avoid dynamo bugs on HPC
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# -------------------------------------------------------
# Usage:
#   sbatch job_annotate_bloom.sh <input_dir> [<pattern>]
#
# Example:
#   sbatch job_annotate_bloom.sh outputs "bloom_input_*.json"
#   sbatch job_annotate_bloom.sh outputs_converted
# -------------------------------------------------------

INPUT_DIR="${1:-outputs}"
PATTERN="${2:-bloom_input_*.json}"

# Judge model path
JUDGE_MODEL="/leonardo_work/EUHPC_D33_215/step_saes/model/Llama-3.3-70B-Instruct"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

if [ ! -d "$JUDGE_MODEL" ]; then
    echo "ERROR: Judge model not found: $JUDGE_MODEL"
    exit 1
fi

echo "Annotating with Bloom's taxonomy: input_dir=$INPUT_DIR pattern=$PATTERN"
echo "Judge model: $JUDGE_MODEL"

python annotate_bloom.py \
    --judge_model_path "$JUDGE_MODEL" \
    --input_dir "$INPUT_DIR" \
    --pattern "$PATTERN" \
    --output_dir outputs_bloom_annotated \
    --batch_size 8 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.90 \
    --save_every 50 \
    --temperature 0.0 \
    --top_p 1.0 \
    --max_new_tokens 8192
