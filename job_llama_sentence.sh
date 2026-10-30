#!/bin/bash
#SBATCH --job-name=llama-sentence
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_llama_sentence_%j.out

module purge
module load conda

source ~/.bashrc
conda activate mental_llm

cd /project2/ruishanl_1185/MentalLLM

python train_critic_sentence.py \
  --base_model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --data_dir /project2/ruishanl_1185/MentalLLM/data \
  --output_dir /project2/ruishanl_1185/MentalLLM/rationale_critic_llama_sentence_checkpoint \
  --results_json final_results.json \
  --log_file train_llama_sentence.out \
  --gpu 0 \
  --attn_impl flash_attention_2 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --max_length 1024 \
  --max_new_tokens 384 \
  --num_epochs 6 \
  --lr 1e-5 \
  --span_weight_cap 30 \
  --span_threshold 0.5 \
  --target_format structured

echo "done"
