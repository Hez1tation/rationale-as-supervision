#!/bin/bash
#SBATCH --job-name=dpo-validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_dpo_validate_%j.out
# Full run: DPO implicit-reward discrimination check on the same 926
# CounselBench-Adv responses used by job_validate_critic.sh. Only run this
# AFTER job_validate_dpo_smoke.sh confirms the adapter path is readable and
# its declared base model matches --base-model below. No generation (just two
# forward passes/response: policy + reference via adapter toggle), so this is
# much faster than job_validate_critic.sh's generation-based run.
#
# Compare the resulting runs/validate_dpo/dpo_summary.json's
# reward_auc_correct_direction against our critic's
# overall_auc_lower_on_failure = 0.549 (runs/critic_validate/summary.json,
# or recompute locally from results/critic_validate responses.jsonl with the
# corrected refinement/validate_critic.py). Same direction convention: 0.5 =
# no discrimination, 1.0 = perfect, <0.5 = backwards.

module purge
module load conda
source ~/.bashrc
conda activate mental_llm

cd /project2/ruishanl_1185/MentalLLM

python -u -m refinement.validate_dpo \
  --input data/counselbench-adv-human-annotation.csv \
  --out runs/validate_dpo \
  --base-model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --device-index 0 \
  --attn-impl flash_attention_2

echo "done"
