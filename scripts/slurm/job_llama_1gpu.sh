#!/bin/bash
#SBATCH --job-name=rcritic-llama-1gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             # 单张任意型号
#SBATCH --constraint="l40s|a40"  # 限定 48G 的 L40S 或 A40,谁空用谁
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_llama_%j.out
# NOTE: SBATCH output 和 --log_file 必须是不同文件——之前都指向 train_llama.out,
# 导致 stdout 和 FileHandler 各写一遍、每行重复两次。slurm_*.out 只兜底
# tqdm/崩溃 traceback 等非 logger 输出,正式日志看 train_llama.out。

module purge
module load conda
source ~/.bashrc
conda activate mental_llm

cd /project2/ruishanl_1185/MentalLLM           # ← 改成你的项目实际路径

# 单卡:直接 python(不要 torchrun)。脚本检测不到 WORLD_SIZE 就走单卡模式。
# 本轮改动:生成目标改 structured(span 引用 + 类型标签 + overall assessment;
# 清洗 na/Nil、横向比较句、no-issue 维度陈述;--target_format plain 可回退对照)。
# max_new_tokens 256->384(结构化输出更长,256 会截断);epoch 10->6(上轮
# 10-epoch 曲线显示 dev loss 第 5 个 epoch 见底,之后单调过拟合)。
python train_critic.py \
  --base_model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --data_dir data \
  --output_dir rationale_critic_llama_checkpoint \
  --results_json final_results.json \
  --log_file train_llama.out \
  --gpu 0 \
  --attn_impl flash_attention_2 \
  --batch_size 2 --grad_accum_steps 4 \
  --max_length 1024 --max_new_tokens 384 \
  --num_epochs 6 --lr 1e-5 \
  --span_weight_cap 30 --span_threshold 0.5 \
  --target_format structured

echo "done"
