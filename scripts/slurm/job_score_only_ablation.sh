#!/bin/bash
#SBATCH --job-name=score-only-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_score_only_%j.out
# Ablation: isolate whether joint span+rationale training is diluting the
# score head under our small (280 unique responses) train set. Same backbone/
# data/hparams as job_llama_sentence.sh (results/Sentence_level_critic), but
# span_loss_weight=0 and rationale_loss_weight=0 -> score-only training.
# Compare this run's test MAE/Spearman/Kendall against:
#   Baseline A (MSE)  : Spearman 0.698 / Kendall 0.558 / MAE 0.660
#   Baseline B (DPO)   : Spearman 0.774 / Kendall 0.735 / MAE 0.571 (*linear-regression-calibrated)
#   Our joint pipeline : Spearman 0.698 / Kendall 0.615 / MAE 0.825
# If this closes most of the MAE gap to Baseline A, the joint objective is the
# problem (fixable via loss reweighting / staged training). If it doesn't,
# the gap is more fundamental (architecture/capacity/backbone choice).

module purge
module load conda

source ~/.bashrc
conda activate mental_llm

cd /project2/ruishanl_1185/MentalLLM

python train_critic_sentence.py \
  --base_model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --data_dir /project2/ruishanl_1185/MentalLLM/data \
  --output_dir /project2/ruishanl_1185/MentalLLM/rationale_critic_llama_sentence_score_only_checkpoint \
  --results_json final_results.json \
  --log_file train_llama_sentence_score_only.out \
  --gpu 0 \
  --attn_impl flash_attention_2 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --max_length 1024 \
  --max_new_tokens 384 \
  --num_epochs 6 \
  --lr 1e-5 \
  --target_format structured \
  --span_loss_weight 0 \
  --rationale_loss_weight 0

echo "done"
