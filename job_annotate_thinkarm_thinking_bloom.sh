#!/bin/bash
#SBATCH --job-name=thinkarm_thinking_bloom
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
mkdir -p output_thinkarm_thinking_bloom

# Better GPU memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable torch.compile to avoid dynamo bugs on HPC
export VLLM_USE_V1=0

echo "Starting ThinkARM thinking trace annotation with Bloom taxonomy..."

# Path to Llama model
LLAMA_PATH="/leonardo_work/EUHPC_D33_215/step_saes/model/Llama-3.3-70B-Instruct"

# Models to annotate (thinking traces only)
MODELS=(
    "deepseekR1"
    "DeepSeek-R1-Distill-Qwen-1.5B"
    "DeepSeek-R1-Distill-Qwen-7B"
    "Phi4R"
)

echo "=========================================="
echo "Annotating thinking traces with Bloom taxonomy..."
echo "=========================================="

python annotate_thinkarm_thinking_bloom.py \
    --judge_model_path "$LLAMA_PATH" \
    --input_dir data/raw \
    --output_dir output_thinkarm_thinking_bloom \
    --models "${MODELS[@]}" \
    --batch_size 16 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.90 \
    --save_every 50

if [ $? -eq 0 ]; then
    echo "✅ All thinking traces annotated!"
else
    echo "❌ Annotation failed!"
    exit 1
fi

echo "=========================================="
echo "Output: output_thinkarm_thinking_bloom/"
echo "=========================================="
