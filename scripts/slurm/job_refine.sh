#!/bin/bash
#SBATCH --job-name=rcritic-refine
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             # 单张任意型号
#SBATCH --constraint="l40s|a40"  # 限定 48G 的 L40S 或 A40,谁空用谁
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                # 同卡要同时装 critic(8B)+generator(8B),内存放宽
#SBATCH --time=05:00:00          # 全量 121 题,~48s/轮、均值 78.5s/q,2.6h 左右,余量到 5h
#SBATCH --output=slurm_refine_%j.out
# Stage 4: critic-guided refinement 批量跑。stdout/tqdm 走 slurm_refine_*.out;
# 结构化结果在 --out 目录下的 traces.jsonl + summary.json。

module purge
module load conda
source ~/.bashrc
conda activate mental_llm

cd /project2/ruishanl_1185/MentalLLM           # ← 改成 refinement/ 和 train_critic.py 所在的项目根

# 全本地、不需要任何外部 API:
#   generator = HF Llama-3.1-8B-Instruct
#   critic    = 本仓库训好的 Rationale-Critic(mode=ours,加载 LoRA adapter + heads.pt)
# critic-base-model 必须和训练时用的同一个 id。critic-code-dir=. 让它能 import train_critic。
# -u: 关闭 stdout 缓冲,进度行 [i/N] 实时刷到日志(否则攒一大批才写,看不到进度)。
# v3 跑:去掉 pre-prompt 里的"质量标准"清单(不再 teaching to the test)。
#   v2(runs/llama_local_v2)= 干净 prompt 但把 critic 的 5 维评分标准明写进生成指令
#     → 每条回答照着考点写,critic 一律打 4-5(std 0.22),refine 无梯度。
#   v3 改 build_initial_prompt:只给"咨询师身份 + 用户消息",不给评分维度(保留输出格式约束)。
#   预期:回答更自然、有好有坏,critic 分数拉开档次(std 变大),refine 才有触发空间。
#   对比 v2 看:mean_initial_score 是否下降、initial 分布 std 是否变大、refinement_rate 是否上升。
python -u -m refinement.run \
  --input data/counselbench-adv-human-annotation.csv \
  --generator hf --gen-model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --gen-device cuda:0 \
  --critic ours \
  --critic-checkpoint rationale_critic_llama_checkpoint/final \
  --critic-base-model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --critic-code-dir . \
  --critic-device-index 0 \
  --dedup-question \
  --max-iter 3 --stop-score 4 \
  --max-new-tokens 512 \
  --out runs/llama_local_v3

echo "done"
