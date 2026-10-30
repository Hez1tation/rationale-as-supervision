#!/bin/bash
#SBATCH --job-name=qwen-sentence
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_qwen_sentence_%j.out

module purge
module load conda

source ~/.bashrc
conda activate mental_llm

cd /project2/ruishanl_1185/MentalLLM

python train_critic_sentence.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --data_dir /project2/ruishanl_1185/MentalLLM/data \
  --output_dir /project2/ruishanl_1185/MentalLLM/rationale_critic_sentence_checkpoint \
  --results_json final_results.json \
  --log_file train_sentence.out \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --max_length 1024 \
  --max_new_tokens 256 \
  --num_epochs 5 \
  --lr 1e-5 \
  --attn_impl flash_attention_2 \
  --span_weight_cap 30 \
  --span_threshold 0.5 \
  --target_format structured

echo "done"
