#!/bin/bash

# Llama-3.1-8B on BOTH L40S via DistributedDataParallel (true parallel compute:
# each GPU holds a full model copy and processes different data; grads are synced).
# Launch with torchrun, NOT plain python. --gpu is ignored under DDP (each rank
# uses its own LOCAL_RANK device). Only rank 0 logs / evaluates / saves.
#
# Per-GPU batch_size=2, grad_accum=2, 2 GPUs  ->  effective batch = 2*2*2 = 8
# (same as the single-GPU bs2/accum4 setup).
#
# Uses the ungated mirror so no gated-access wait. Swap to
# meta-llama/Llama-3.1-8B-Instruct once your access request is approved.

nohup torchrun --standalone --nproc_per_node=2 \
  train_critic.py \
  --base_model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --data_dir data \
  --output_dir rationale_critic_llama_checkpoint \
  --results_json final_results.json \
  --log_file train_llama.out \
  --attn_impl flash_attention_2 \
  --batch_size 2 \
  --grad_accum_steps 2 \
  --max_length 1024 \
  --max_new_tokens 256 \
  --num_epochs 5 \
  --lr 1e-5 \
  > train_llama_console.out 2>&1 &

echo "DDP training started (Llama-3.1-8B on 2x L40S)."
echo "Log file: train_llama.out"
echo "Results:  rationale_critic_llama_checkpoint/final_results.json"
