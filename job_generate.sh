#!/bin/bash
#SBATCH --job-name=thinkarm_gen
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

mkdir -p logs outputs

# Disable torch.compile to avoid dynamo bugs on HPC
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# -------------------------------------------------------
# Usage:
#   sbatch job_generate.sh <model_name> <dataset_name>
#
# Example:
#   sbatch job_generate.sh qwen3_30b_thinking gsm8k
#   sbatch job_generate.sh deepseek_r1_distill_qwen_7b math
#   sbatch job_generate.sh phi_4_reasoning gsm_hard
# -------------------------------------------------------

MODEL_NAME="${1:-qwen3_30b_thinking}"
DATASET_NAME="${2:-gsm8k}"

# Model paths
declare -A MODEL_PATHS
MODEL_PATHS[qwen3_30b_thinking]="/leonardo_work/EUHPC_D33_216/mzoumpou/model/Qwen3-30b-thinking"
MODEL_PATHS[qwen3_4b_thinking]="/leonardo_work/EUHPC_D33_216/mzoumpou/model/Qwen3-4b-thinking"
MODEL_PATHS[phi_4_reasoning]="/leonardo_work/EUHPC_D33_216/mzoumpou/model/phi_4_reasoning"
MODEL_PATHS[deepseek_r1_distill_llama8b]="/leonardo_work/EUHPC_D33_216/mzoumpou/model/deepseek_r1_distill_llama8b"
MODEL_PATHS[deepseek_r1_distill_qwen_1.5b]="/leonardo_work/EUHPC_D33_216/mzoumpou/model/deepseek_r1_distill_qwen_1.5b"
MODEL_PATHS[deepseek_r1_distill_qwen_7b]="/leonardo_work/EUHPC_D33_216/mzoumpou/model/deepseek_r1_distill_qwen_7b"

# Dataset paths
declare -A DATASET_PATHS
DATASET_PATHS[gsm8k]="/leonardo_work/EUHPC_D33_216/mzoumpou/datasets/gsm8k.json"
DATASET_PATHS[gsm_hard]="/leonardo_work/EUHPC_D33_216/mzoumpou/datasets/gsm_hard.json"
DATASET_PATHS[math]="/leonardo_work/EUHPC_D33_216/mzoumpou/datasets/math.json"

MODEL_PATH="${MODEL_PATHS[$MODEL_NAME]}"
DATASET_PATH="${DATASET_PATHS[$DATASET_NAME]}"

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: Unknown model '$MODEL_NAME'"
    echo "Options: qwen3_30b_thinking, qwen3_4b_thinking, phi_4_reasoning, deepseek_r1_distill_llama8b, deepseek_r1_distill_qwen_1.5b, deepseek_r1_distill_qwen_7b"
    exit 1
fi

if [ -z "$DATASET_PATH" ]; then
    echo "ERROR: Unknown dataset '$DATASET_NAME'"
    echo "Options: gsm8k, gsm_hard, math"
    exit 1
fi

# Per-model tensor-parallel size
TP_SIZE=4
case "$MODEL_NAME" in
    phi_4_reasoning) TP_SIZE=2 ;;
esac

echo "Generating thinking traces: model=$MODEL_NAME dataset=$DATASET_NAME tp=$TP_SIZE"

python generate.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --dataset_name "$DATASET_NAME" \
    --output_dir outputs \
    --save_every 50 \
    --tensor_parallel_size "$TP_SIZE" \
    --gpu_memory_utilization 0.90 \
    --batch_size 32 \
    --max_new_tokens 2048
