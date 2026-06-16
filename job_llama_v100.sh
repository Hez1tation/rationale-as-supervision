#!/bin/bash
#SBATCH --job-name=rcritic-llama-v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2        # 需要 32GB 版 V100;16GB 版放不下 8B
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00          # V100 比 L40S 慢,时间给足
#SBATCH --output=slurm_llama_v100_%j.out

module purge
module load conda
source ~/.bashrc
conda activate mental_llm

cd /home1/jitianmi/...           # ← 改成你的项目实际路径

# V100: 必须 --fp16(无 bf16);32GB 较紧 -> 开 --grad_checkpoint + batch_size 1
torchrun --standalone --nproc_per_node=2 \
  train_critic.py \
  --base_model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --data_dir data \
  --output_dir rationale_critic_llama_checkpoint \
  --results_json final_results.json \
  --log_file train_llama.out \
  --fp16 \
  --grad_checkpoint \
  --batch_size 1 --grad_accum_steps 4 \
  --max_length 1024 --max_new_tokens 256 \
  --num_epochs 5 --lr 1e-5

echo "done"
