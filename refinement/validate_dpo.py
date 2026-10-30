"""Validate the DPO baseline's implicit reward against CounselBench-Adv human labels.

Why this exists
----------------
``validate_critic.py`` showed our critic's ``overall`` score barely
discriminates human-flagged-bad responses from clean ones on 926 real
CounselBench-Adv responses (``overall_auc_lower_on_failure`` ~0.55, near
chance). Before concluding that is a rationale-critic-specific failure, we
need the SAME test run against the DPO baseline
(``refinement/generator.py``'s ``DEFAULT_DPO_ADAPTER``) on the IDENTICAL set
of responses and labels:

  * If DPO's implicit reward ALSO collapses to ~chance on this out-of-
    distribution data -> the story is "no scorer trained on this ~100-
    question dataset generalizes OOD", not "our critic specifically fails"
    -- and only the rationale/span critic can tell you *where* it fails.
  * If DPO's reward still discriminates fine OOD -> the gap is real and
    specific to this critic's objective/architecture, not just data size.

Method
------
DPO has no separate reward head; its trained POLICY implicitly defines a
reward via the standard DPO identity::

    r(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))

pi_theta = base model + the trained DPO LoRA adapter; pi_ref = the same base
model with the adapter disabled (``PeftModel.disable_adapter()``). Both are
served from ONE loaded 8B model, not two.

beta is a fixed positive training hyperparameter this repo does not own, and
it cancels out of every RANK-based comparison (Spearman/Kendall/AUC). We
report ``log_ratio = log pi_theta(y|x) - log pi_ref(y|x))`` (summed over
response tokens, matching DPO's own training objective) as the reward proxy.
This script answers "does DPO's reward rank real good/bad responses
correctly OOD", NOT "what is DPO's calibrated MAE" (that additionally needs
beta and a linear fit to gold scores -- a separate, later ask if this
discrimination check comes back positive).

Grouping / labels reuse ``load_grouped_responses()`` from validate_critic.py,
so the (post,response) set and mode_labels are IDENTICAL to the critic run
-- that is what makes the AUC numbers directly comparable.

DPO's reward is a single scalar (no empathy/specificity/factual/toxicity
breakdown like our critic has), so this script only reproduces the
``overall_auc_lower_on_failure``-style "any failure present" comparison, not
the full per-mode table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from refinement.validate_critic import _auc_lower, _mean, _std, detect_columns, load_grouped_responses

DEFAULT_DPO_ADAPTER = Path(
    "/project2/ruishanl_1185/rationale_as_supervision/DPO/model/checkpoints/"
    "dpo_overall_only_aggregate_4gpu_100e"
)
DEFAULT_BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"


def build_prompt(question: str) -> str:
    """Mirrors refinement/prompts.py build_initial_prompt's counselor framing
    (minus OUTPUT_FORMAT_RULES, which only govern generation formatting and
    are irrelevant when scoring an already-written response)."""
    return (
        "You are a mental-health counselor. Someone has reached out to you for support.\n\n"
        "Their message:\n"
        f"{question.strip()}\n\n"
        "Write your response to them."
    )


class DPOScorer:
    def __init__(self, base_model: str, adapter_dir: str | Path, device_index: int = 0,
                 attn_impl: str = "flash_attention_2"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
        ).to(self.device)
        self.model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        self.model.eval()
        self.boundary_mismatches = 0

    def _seq_logprob(self, prompt_ids, full_ids) -> float | None:
        """Sum of log p(token_t | tokens_<t) over the response-token span."""
        torch = self.torch
        if full_ids.shape[1] <= prompt_ids.shape[1]:
            return None  # response tokenized to nothing

        # Tokenizing "prompt" and "prompt+response" separately can drift at the
        # boundary (BPE merges). Fall back to the actual longest common prefix
        # so the response span is measured from where the token streams truly
        # diverge, not from the (possibly wrong) standalone prompt length.
        prompt_len = prompt_ids.shape[1]
        max_check = min(prompt_len, full_ids.shape[1])
        if not torch.equal(full_ids[0, :max_check], prompt_ids[0, :max_check]):
            self.boundary_mismatches += 1
            common = 0
            for k in range(max_check):
                if full_ids[0, k].item() == prompt_ids[0, k].item():
                    common = k + 1
                else:
                    break
            prompt_len = common
            if full_ids.shape[1] <= prompt_len:
                return None

        with torch.inference_mode():
            out = self.model(input_ids=full_ids)
            logits = out.logits[:, :-1, :]
            targets = full_ids[:, 1:]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            token_logp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
        start = prompt_len - 1  # token_logp[j] predicts full_ids[j+1]
        return float(token_logp[0, start:].sum().item())

    def score(self, question: str, response: str) -> dict[str, float | None]:
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(question)}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        full_ids = self.tokenizer(prompt_text + response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        policy_logp = self._seq_logprob(prompt_ids, full_ids)
        with self.model.disable_adapter():
            ref_logp = self._seq_logprob(prompt_ids, full_ids)
        if policy_logp is None or ref_logp is None:
            return {"policy_logp": policy_logp, "ref_logp": ref_logp, "log_ratio": None}
        return {"policy_logp": policy_logp, "ref_logp": ref_logp, "log_ratio": policy_logp - ref_logp}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    yes, no = [], []
    for r in results:
        lr = r.get("log_ratio")
        if lr is None:
            continue
        any_present = any(r["mode_labels"].values())
        (yes if any_present else no).append(float(lr))
    # log_ratio: higher = the DPO policy prefers this response more (= "better").
    # So a human-flagged-bad response is expected to score LOWER -> same
    # direction as our critic's non-reversed dims -> _auc_lower(yes, no) is
    # already the correct-direction test (no reversal needed here).
    return {
        "n_responses": len(results),
        "n_any_failure": len(yes),
        "n_no_failure": len(no),
        "reward_mean_any_failure": _mean(yes),
        "reward_mean_no_failure": _mean(no),
        "reward_std_any_failure": _std(yes),
        "reward_std_no_failure": _std(no),
        "reward_auc_correct_direction": _auc_lower(yes, no),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CounselBench-Adv CSV (same file as validate_critic.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--dpo-adapter", default=str(DEFAULT_DPO_ADAPTER))
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--attn-impl", default="flash_attention_2")
    ap.add_argument("--limit", type=int, default=None, help="cap #responses (smoke test)")
    ap.add_argument("--dump-schema", action="store_true",
                     help="check adapter path + column mapping, print label counts, then exit (no model load)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")
    with in_path.open("r", encoding="utf-8-sig", newline="") as handle:
        fieldnames = (csv.DictReader(handle).fieldnames) or []
    cols = detect_columns(fieldnames)
    print(f"[schema] CSV columns: {fieldnames}", flush=True)
    print(f"[schema] mapped fields: {cols}", flush=True)
    missing = [k for k in ("post", "response", "mode", "human") if not cols[k]]
    if missing:
        sys.exit(f"Could not find columns for {missing}.")

    records = load_grouped_responses(in_path, cols)
    print(f"[data] {len(records)} unique (post,response) responses", flush=True)

    adapter_dir = Path(args.dpo_adapter)
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        sys.exit(
            f"DPO adapter not found or unreadable: {adapter_dir} "
            f"(expected adapter_config.json there -- check the path / your read "
            f"permission on this labmate-owned checkpoint on CARC)"
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    declared_base = cfg.get("base_model_name_or_path")
    print(f"[dpo] adapter_config.json declares base_model_name_or_path={declared_base!r}; "
          f"script will load --base-model={args.base_model!r} -- confirm these refer to the "
          f"same checkpoint (org/name can differ, e.g. NousResearch mirror vs meta-llama) "
          f"before trusting the results.", flush=True)

    if args.dump_schema:
        return

    if args.limit:
        records = records[: args.limit]
        print(f"[data] limited to {len(records)} responses", flush=True)

    scorer = DPOScorer(args.base_model, adapter_dir, device_index=args.device_index, attn_impl=args.attn_impl)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dpo_responses.jsonl"

    results: list[dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for i, rec in enumerate(records, 1):
            sc = scorer.score(rec["post"], rec["response"])
            row = {"id": rec["id"], "model_name": rec["model_name"], "mode_labels": rec["mode_labels"], **sc}
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{i}/{len(records)}] {rec['id']} model={rec['model_name']} "
                  f"log_ratio={sc['log_ratio']}", flush=True)

    if scorer.boundary_mismatches:
        print(f"[warn] {scorer.boundary_mismatches}/{len(records)} examples had a "
              f"prompt/full tokenization boundary mismatch (handled via longest-common-prefix "
              f"fallback); this adds minor noise but should not bias direction.", flush=True)

    summary = summarize(results)
    (out_dir / "dpo_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n==== DPO SUMMARY ====", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"\nResponses -> {out_path}", flush=True)
    print(f"Summary   -> {out_dir / 'dpo_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
