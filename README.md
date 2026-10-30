# Rationale-as-Supervision: a Rationale-Critic for subjective-safety alignment

Training code for a **Rationale-Critic**: instead of compressing expert judgment into a
scalar reward, we supervise a critic model with experts' **natural-language rationales**
and **span-level failure annotations**, and test whether this yields better, more
sample-efficient alignment on subjective-safety tasks (mental-health QA) than a scalar
reward model or DPO trained on the same data.

The critic uses **one backbone (Llama-3.1-8B-Instruct + LoRA)** with three signals:
- score regression head (multi-dimension: overall / empathy / specificity / factual / toxicity),
- span head (multi-label sigmoids: medical / factual / toxicity),
- native LM head for **rationale (critique) generation**.

See `docs/TRAINING_WRITEUP.md` for the full architecture / data / loss design and known limitations.

> **Data is not included.** This repo is code-only. The CounselBench expert annotations
> are managed separately and are not committed.

## Layout

| Path | Purpose |
|------|---------|
| `train_critic.py` | Main token-level critic training entry point: joint score + span + rationale losses |
| `train_critic_sentence.py` | Sentence-level span-label variant for direct comparison with `train_critic.py` |
| `refinement/` | Stage 4 critic-guided generation, revision, and critic validation code |
| `scripts/preprocess/` | Data preparation utilities, including `prepare_spans.py` |
| `scripts/eval/` | Offline evaluation, target checks, and LLM-judge sample dumping |
| `scripts/local/` | Local/background launch scripts |
| `scripts/slurm/` | CARC/SLURM job scripts |
| `data/` | Expected local data directory, not committed |
| `results/` | Stage 4 packaged experiment outputs |
| `artifacts/` | Historical outputs, judge samples, and local archives |
| `docs/` | Longer writeups and experiment notes |

## Running

Training runs on a GPU server, typically 1-2 L40S / A100 GPUs. Example launch:

```bash
bash scripts/local/train_llama_ddp.sh
```

Expected data layout (not committed): `data/{train,dev,test}.jsonl`, each row augmented
with `span_labels` by `scripts/preprocess/prepare_spans.py`.
