#!/bin/bash

# Llama-3.1-8B-Instruct variant. Same script/data as the Qwen run; only the base
# model + output paths differ, so Qwen results are not overwritten.
# NOTE: meta-llama/Llama-3.1-8B-Instruct is GATED on HuggingFace. If your access
# request is still "awaiting review", either wait for approval or use the ungated
# mirror NousResearch/Meta-Llama-3.1-8B-Instruct (same weights).
#
# Single L40S (48GB): whole model on one GPU, FlashAttention-2 on. Set --gpu 1 to
# run this on the second card while Qwen runs on gpu 0.

nohup python train_critic.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --data_dir data \
  --output_dir rationale_critic_llama_checkpoint \
  --results_json final_results.json \
  --log_file train_llama.out \
  --gpu 0 \
  --attn_impl flash_attention_2 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --max_length 1024 \
  --max_new_tokens 384 \
  --num_epochs 6 \
  --lr 1e-5 \
  --target_format structured \
  > train_llama_console.out 2>&1 &

echo "Training started (Llama-3.1-8B-Instruct on gpu 0)."
echo "Log file: train_llama.out"
echo "Results:  rationale_critic_llama_checkpoint/final_results.json"
