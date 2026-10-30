#!/bin/bash
#SBATCH --job-name=stage5-lite
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_stage5_lite_%j.out

set -eo pipefail

module purge
module load conda

source ~/.bashrc
conda activate mental_llm

cd /project2/ruishanl_1185/MentalLLM

# Stage5-lite defaults:
#   LIMIT=40      small pilot; use LIMIT=120 for the full Adv set
#   MAX_ITER=1    safer first-pass baseline; use MAX_ITER=3 for the schedule version
#   STOP_SCORE=4  same stopping rule as Stage 4
LIMIT="${LIMIT:-40}"
MAX_ITER="${MAX_ITER:-1}"
STOP_SCORE="${STOP_SCORE:-4}"
OUT_DIR="${OUT_DIR:-runs/stage5_lite_sentence}"
INPUT="${INPUT:-data/counselbench-adv-human-annotation.csv}"
SEED="${SEED:-42}"

if [ "${LIMIT}" = "all" ]; then
  LIMIT_ARGS=()
else
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

mkdir -p "${OUT_DIR}"

echo "Stage5-lite starting"
echo "INPUT=${INPUT}"
echo "OUT_DIR=${OUT_DIR}"
echo "LIMIT=${LIMIT}"
echo "MAX_ITER=${MAX_ITER}"
echo "STOP_SCORE=${STOP_SCORE}"
echo "SEED=${SEED}"

python -u -m refinement.run \
  --input "${INPUT}" \
  --dedup-question \
  "${LIMIT_ARGS[@]}" \
  --generator hf \
  --gen-model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --gen-device cuda:0 \
  --critic ours \
  --critic-checkpoint rationale_critic_llama_sentence_checkpoint/final \
  --critic-base-model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --critic-code-dir . \
  --critic-device-index 0 \
  --max-iter "${MAX_ITER}" \
  --stop-score "${STOP_SCORE}" \
  --quality-gate \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --top-p 0.9 \
  --out "${OUT_DIR}"

python stage5_lite_make_ab.py \
  --traces "${OUT_DIR}/traces.jsonl" \
  --out "${OUT_DIR}/blind_ab.jsonl" \
  --key-out "${OUT_DIR}/blind_ab_key.jsonl" \
  --summary-out "${OUT_DIR}/stage5_lite_summary.json" \
  --seed "${SEED}"

echo "Stage5-lite done"
echo "Trace summary: ${OUT_DIR}/summary.json"
echo "Blind A/B file: ${OUT_DIR}/blind_ab.jsonl"
echo "Private A/B key: ${OUT_DIR}/blind_ab_key.jsonl"
echo "Pilot summary: ${OUT_DIR}/stage5_lite_summary.json"
