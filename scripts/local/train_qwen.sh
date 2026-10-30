#!/bin/bash

# Single L40S (48GB) — whole model on one GPU, no device_map=auto, no grad
# checkpointing (plenty of memory), FlashAttention-2 on. Set --gpu 1 to use the
# second card instead (e.g. run a Llama job on gpu 0 and Qwen on gpu 1).

cd "$(dirname "$0")/../.."

nohup python train_critic.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --data_dir data \
  --output_dir rationale_critic_checkpoint \
  --gpu 0 \
  --attn_impl flash_attention_2 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --max_length 1024 \
  --max_new_tokens 256 \
  --num_epochs 5 \
  --lr 1e-5 \
  > train_console.out 2>&1 &

echo "Training started (Qwen2.5-7B on gpu 0)."
echo "Log file: train.out"
echo "Results:  rationale_critic_checkpoint/final_results.json"
