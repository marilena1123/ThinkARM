#!/bin/bash
#SBATCH --job-name=thinkarm_generic
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --account=EUHPC_D33_235
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load cuda/12.2
module load anaconda3/2023.09-0

source activate /leonardo_work/EUHPC_D33_216/mzoumpou/cogchains_env

cd /leonardo_work/EUHPC_D33_216/mzoumpou/ThinkARM

mkdir -p logs
mkdir -p output_thinkarm_our_datasets

# Disable torch.compile to avoid dynamo bugs on HPC
export VLLM_USE_V1=0

# Better GPU memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting ThinkARM annotation for multiple datasets..."

# Path to Llama model
LLAMA_PATH="/leonardo_work/EUHPC_D33_215/step_saes/model/Llama-3.3-70B-Instruct"

# Models to annotate
MODELS=(
    "deepseek_r1"
    "deepseek_r1_distill_llama8b"
    "deepseek_r1_distill_qwen1_5b"
    "deepseek_r1_distill_qwen7b"
    "phi_4_reasoning"
    "qwen3_4b_thinking"
    "qwen3_30b_thinking"
)

# Datasets
DATASETS=("gsm8k" "gsm_hard" "math")

# Base input directory
INPUT_BASE="/leonardo_work/EUHPC_D33_216/mzoumpou/outputs_full_default_params"

# Annotate each model and dataset
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Processing model: $MODEL"
    echo "=========================================="

    for DATASET in "${DATASETS[@]}"; do
        INPUT_FILE="$INPUT_BASE/$MODEL/cot_${MODEL}_${DATASET}.json"
        OUTPUT_DIR="output_thinkarm_our_datasets/$MODEL/$DATASET"

        if [ ! -f "$INPUT_FILE" ]; then
            echo "⚠️  File not found: $INPUT_FILE"
            continue
        fi

        echo "  Annotating: $DATASET"

        python annotate_thinkarm_generic.py \
            --input_file "$INPUT_FILE" \
            --judge_model_path "$LLAMA_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --reasoning_field thinking_trace \
            --question_field question

        if [ $? -eq 0 ]; then
            echo "  ✅ Completed: $DATASET"
        else
            echo "  ❌ Failed: $DATASET"
        fi

        echo ""
    done
done

echo "=========================================="
echo "All datasets annotated!"
echo "Output: output_thinkarm_our_datasets/"
echo "=========================================="
