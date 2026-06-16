# Rationale-as-Supervision: a Rationale-Critic for subjective-safety alignment

Training code for a **Rationale-Critic**: instead of compressing expert judgment into a
scalar reward, we supervise a critic model with experts' **natural-language rationales**
and **span-level failure annotations**, and test whether this yields better, more
sample-efficient alignment on subjective-safety tasks (mental-health QA) than a scalar
reward model or DPO trained on the same data.

The critic uses **one backbone (Llama-3.1-8B-Instruct + LoRA)** with three signals:
- score regression head (multi-dimension: overall / empathy / specificity / factual / toxicity),
- token-level **span head** (multi-label sigmoids: medical / factual / toxicity),
- native LM head for **rationale (critique) generation**.

See `TRAINING_WRITEUP.md` for the full architecture / data / loss design and known limitations.

> **Data is not included.** This repo is code-only. The CounselBench expert annotations
> are managed separately and are not committed.

## Layout

| File | Purpose |
|------|---------|
| `train_critic.py` | Main training script — one backbone, joint score + span + rationale losses |
| `prepare_spans.py` | Data prep: resolve real failure spans from CounselBench `*_copy` fields into `span_labels` |
| `eval_offline.py` | Offline evaluation over an exported `final_results.json` |
| `check_targets.py` | Sanity checks on training targets |
| `dump_judge_sample.py` | Dump a sample for LLM-judge inspection |
| `train_llama*.sh`, `job_llama*.sh` | Launch / SLURM scripts (Llama backbone, single-GPU & DDP) |
| `train_qwen.sh`, `job_qwen.sh` | Launch scripts for the Qwen backbone variant |

## Running

Training runs on a GPU server (1–2× L40S / A100). Example single-GPU launch:

```bash
bash train_llama.sh
```

Expected data layout (not committed): `data/{train,dev,test}.jsonl`, each row augmented
with `span_labels` by `prepare_spans.py`.
