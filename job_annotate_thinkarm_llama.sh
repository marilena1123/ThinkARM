#!/bin/bash
#SBATCH --job-name=thinkarm_llama
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --account=EUHPC_D33_235
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load cuda/12.2
module load anaconda3/2023.09-0

source activate /leonardo_work/EUHPC_D33_216/mzoumpou/cogchains_env

cd /leonardo_work/EUHPC_D33_216/mzoumpou/ThinkARM

mkdir -p logs
mkdir -p output_thinkarm_llama

# Disable torch.compile to avoid dynamo bugs on HPC
export VLLM_USE_V1=0

echo "Starting ThinkARM annotation with Llama judge..."

# Path to Llama model
LLAMA_PATH="/leonardo_work/EUHPC_D33_216/mzoumpou/models/Llama-3.3-70B-Instruct"

# Models to annotate
MODELS=("deepseekR1" "DeepSeek-R1-Distill-Qwen-7B" "DeepSeek-R1-Distill-Qwen-1.5B" "Phi4R")

# Annotate each model
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Annotating: $MODEL"
    echo "=========================================="

    python annotate_thinkarm_llama.py \
        --response_model "$MODEL" \
        --judge_model_path "$LLAMA_PATH"

    if [ $? -eq 0 ]; then
        echo "✅ Completed: $MODEL"
    else
        echo "❌ Failed: $MODEL"
    fi

    echo ""
done

echo "=========================================="
echo "All models annotated!"
echo "Output: output_thinkarm_llama/"
echo "=========================================="
