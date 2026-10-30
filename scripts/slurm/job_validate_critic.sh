#!/bin/bash
#SBATCH --job-name=rcritic-validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|a40"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                # 只装 critic(8B),不需要 generator
#SBATCH --time=04:00:00          # ~1000 条回答 × 每条几秒,1-2h,余量到 4h
#SBATCH --output=slurm_validate_%j.out
# 验证实验:把 critic 跑在 CounselBench-Adv 自带的 new_response(9 个模型真实回答)上,
# 对照 human_answer 的逐 mode 失败标注,判断 critic 在【已知好坏】的回答上能不能分辨。
#   结果:critic 对人工标"有问题"的回答是否给更低分 / 标出匹配 span。
#   能分辨 -> Stage4 饱和是 generator 太干净的锅;不能分辨 -> critic 本身要重训。

module purge
module load conda
source ~/.bashrc
conda activate mental_llm

cd /project2/ruishanl_1185/MentalLLM           # ← refinement/ 和 train_critic.py 所在的项目根

# 先冒烟自检:确认列名映射对、各 mode 标签数量合理(不跑模型,秒回)。
# 正式跑前强烈建议先单独执行这一行看输出:
#   python -u -m refinement.validate_critic --input data/counselbench-adv-human-annotation.csv \
#     --out runs/critic_validate --critic-checkpoint x --critic-base-model x --dump-schema
# critic-checkpoint / base-model 必须和 Stage 4 用的同一套。
python -u -m refinement.validate_critic \
  --input data/counselbench-adv-human-annotation.csv \
  --out runs/critic_validate \
  --critic-checkpoint rationale_critic_llama_checkpoint/final \
  --critic-base-model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --critic-code-dir . \
  --critic-device-index 0 \
  --max-new-tokens 256

echo "done"
