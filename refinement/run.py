"""Batch runner for the Stage 4 critic-guided refinement pipeline.

Wires the existing pieces together end-to-end:
    load_prompt_records -> critic_guided_generate -> write_jsonl + summarize_traces

Examples
--------
Local dry-run (no GPU, no API) -- validates the whole loop with mocks:

    python -m refinement.run \
        --input data/counselbench_adv.csv \
        --generator mock --critic mock \
        --out runs/dryrun --limit 5

Server, fully local (no external API): HF Llama generator + our trained critic:

    python -m refinement.run \
        --input data/counselbench_adv.csv \
        --generator hf --gen-model meta-llama/Llama-3.1-8B-Instruct \
        --critic ours \
        --critic-checkpoint rationale_critic_llama_checkpoint/final \
        --critic-base-model NousResearch/Meta-Llama-3.1-8B-Instruct \
        --critic-code-dir . \
        --out runs/llama_local

Server, GPT-4o generator (needs OPENAI_API_KEY in .env):

    python -m refinement.run \
        --input data/counselbench_adv.csv \
        --generator openai --gen-model gpt-4o \
        --critic ours \
        --critic-checkpoint rationale_critic_llama_checkpoint/final \
        --critic-base-model NousResearch/Meta-Llama-3.1-8B-Instruct \
        --critic-code-dir . \
        --out runs/gpt4o
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .critic import build_critic
from .generator import DEFAULT_BASE_MODEL, build_generator
from .io import load_prompt_records, summarize_traces
from .refine import RefinementConfig, critic_guided_generate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run critic-guided refinement over a prompt file and write traces + summary.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Prompt source: a .jsonl / .csv file path, or hf:<repo_id> to pull a "
        "CounselBench-Adv dataset from the HuggingFace Hub "
        "(e.g. hf:izi-ano/CounselBench-Adv-human-annotation)",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory (traces.jsonl + summary.json written here)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N prompts")
    parser.add_argument(
        "--dedup-question",
        action="store_true",
        help="Keep only the first record per unique question text (e.g. test.jsonl has each "
        "question x4 responders; Stage 4 only uses the question, so this drops 80 rows to 20).",
    )

    # Generator
    parser.add_argument("--generator", choices=["mock", "hf", "openai"], default="mock")
    parser.add_argument("--gen-model", default=DEFAULT_BASE_MODEL, help="Generator model name / HF id / OpenAI model")
    parser.add_argument("--gen-adapter", default=None, help="Optional PEFT adapter dir for the HF generator (e.g. a DPO adapter)")
    parser.add_argument("--gen-4bit", action="store_true", help="Load the HF generator in 4-bit")
    parser.add_argument("--gen-device", default=None, help="Device for the HF generator, e.g. cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    # Critic
    parser.add_argument("--critic", choices=["mock", "ours", "local", "pipeline"], default="mock")
    parser.add_argument("--critic-checkpoint", default=None, help="save_checkpoint dir (mode=ours) or pipeline checkpoint")
    parser.add_argument("--critic-base-model", default=None, help="HF id used for training the critic (mode=ours)")
    parser.add_argument("--critic-code-dir", default=None, help="Dir holding train_critic.py (ours) or stage3 scripts (pipeline)")
    parser.add_argument("--critic-device-index", type=int, default=0, help="GPU index for mode=ours critic")
    parser.add_argument("--critic-4bit", action="store_true", help="Load the critic in 4-bit (mode=pipeline)")

    # Refinement loop
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--stop-score", type=int, default=4)
    parser.add_argument("--require-no-issue-text", action="store_true", help="Stricter AND stop: score high AND critic states no issues")
    parser.add_argument("--quality-gate", action="store_true", help="Enable the deterministic formatting quality gate")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    records = load_prompt_records(args.input)
    if args.dedup_question:
        seen: set[str] = set()
        deduped: list[dict] = []
        for record in records:
            key = " ".join(record["question"].split()).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        print(f"Deduped questions: {len(records)} -> {len(deduped)} records")
        records = deduped
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print(f"No prompt records found in {args.input}")
        return 1

    generator = build_generator(
        args.generator,
        model_name=args.gen_model,
        adapter_dir=args.gen_adapter,
        device=args.gen_device,
        load_in_4bit=args.gen_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    critic = build_critic(
        args.critic,
        checkpoint=args.critic_checkpoint,
        code_dir=args.critic_code_dir,
        base_model=args.critic_base_model,
        device_index=args.critic_device_index,
        load_in_4bit=args.critic_4bit,
        max_new_tokens=args.max_new_tokens,
    )
    config = RefinementConfig(
        max_iter=args.max_iter,
        stop_score=args.stop_score,
        require_no_issue_text=args.require_no_issue_text,
        enable_quality_gate=args.quality_gate,
    )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "traces.jsonl"
    summary_path = out_dir / "summary.json"

    # Write each trace as it completes so a killed/timed-out run still keeps its
    # finished work (the loop is long; a wall-clock kill would otherwise lose
    # everything). Summary is recomputed from the in-memory list at the end.
    traces: list[dict] = []
    elapsed_total = 0.0
    with traces_path.open("w", encoding="utf-8", newline="\n") as handle:
        for i, record in enumerate(records, 1):
            t0 = time.perf_counter()
            trace = critic_guided_generate(record["question"], generator, critic, config)
            dt = time.perf_counter() - t0
            elapsed_total += dt
            # carry the input identity alongside the refinement trace
            trace = {
                "id": record.get("id"),
                "target_failure_mode": record.get("target_failure_mode"),
                "question": record["question"],
                **trace,
            }
            traces.append(trace)
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{i}/{len(records)}] {record.get('id')}  "
                f"final_score={trace.get('final_score')}  stop={trace.get('stop_reason')}  "
                f"iters={len(trace.get('iterations') or [])}  "
                f"{dt:.1f}s (avg {elapsed_total / i:.1f}s/q)",
                flush=True,
            )

    summary = summarize_traces(traces)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(traces)} traces -> {traces_path}")
    print(f"Summary -> {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
