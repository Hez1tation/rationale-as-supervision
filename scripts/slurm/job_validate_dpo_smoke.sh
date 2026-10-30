#!/bin/bash
#SBATCH --job-name=dpo-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_dpo_smoke_%j.out
# Smoke test for refinement/validate_dpo.py -- run this FIRST, before the full
# 926-response job. Checks: can we even read the DPO adapter directory (it's a
# labmate-owned checkpoint, permissions unconfirmed), does its declared base
# model match ours, and does the CSV column mapping look right. No model load,
# seconds to run.

module purge
module load conda
source ~/.bashrc
conda activate mental_llm

cd /project2/ruishanl_1185/MentalLLM

python -u -m refinement.validate_dpo \
  --input data/counselbench-adv-human-annotation.csv \
  --out runs/validate_dpo \
  --dump-schema

echo "done"
