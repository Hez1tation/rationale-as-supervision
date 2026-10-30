#!/bin/bash

# ===== Conda =====

module purge
module load conda

source ~/.bashrc
conda activate mental_llm

# ===== Project Dir =====

cd /project2/ruishanl_1185/MentalLLM

# ===== Run Training =====

nohup python train_critic.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --data_dir /project2/ruishanl_1185/MentalLLM/data \
  --output_dir /project2/ruishanl_1185/MentalLLM/rationale_critic_checkpoint \
  --batch_size 1 \
  --max_length 1024 \
  --max_new_tokens 256 \
  --num_epochs 5 \
  --lr 1e-5 > train_console.out 2>&1 &

echo "Training started."
echo "Log file: train.out"
echo "Results:  rationale_critic_checkpoint/final_results.json"

# ===== Monitor =====

# tail -f train.out
