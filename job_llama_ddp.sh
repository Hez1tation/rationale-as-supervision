#!/bin/bash
#SBATCH --job-name=rcritic-llama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2        # 点名 2 张 L40S(48GB×2);改 a40/a100 也行
#SBATCH --nodes=1
#SBATCH --ntasks=1               # torchrun 自己 spawn 2 个进程,这里保持 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                # 主机内存(非显存)
#SBATCH --time=04:00:00
#SBATCH --output=slurm_llama_ddp_%j.out  # 与 --log_file 分开,避免每行日志重复两次

module purge
module load conda
source ~/.bashrc
conda activate mental_llm

cd /project2/ruishanl_1185/MentalLLM   # ← 改成你的项目实际路径

# DDP 两卡;sbatch 里不要 nohup、不要 & 、不要 > 重定向(SBATCH --output 已接管)
torchrun --standalone --nproc_per_node=2 \
  train_critic.py \
  --base_model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --data_dir data \
  --output_dir rationale_critic_llama_checkpoint \
  --results_json final_results.json \
  --log_file train_llama.out \
  --attn_impl flash_attention_2 \
  --batch_size 2 --grad_accum_steps 2 \
  --max_length 1024 --max_new_tokens 256 \
  --num_epochs 5 --lr 1e-5

echo "done"
