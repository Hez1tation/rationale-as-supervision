"""Validate the trained critic against CounselBench-Adv human failure labels.

Why this exists
---------------
Stage 4 showed the critic scores nearly every Llama-8B response 4-5, so the
refinement loop rarely fires. That has two competing explanations we cannot
separate from the Stage 4 run alone:

  (a) the critic is saturated / cannot discriminate (a critic problem), OR
  (b) the Llama-8B outputs are genuinely clean, so high scores are correct
      and the real fix is a stronger generator (a generator problem).

This script disambiguates by running the critic on responses whose quality we
ALREADY know: CounselBench-Adv ships `new_response` (9 models' actual answers)
with per-failure-mode human labels (`human_answer` = is this failure present?).
We feed those known responses to the critic and check whether it separates the
human-flagged-bad cases from the rest.

Verdict logic
-------------
  * If the critic gives clearly LOWER scores (and/or flags matching spans) on
    human-flagged-bad responses -> the critic CAN discriminate -> Stage 4
    saturation is the generator's fault (outputs really are clean).
  * If the critic scores bad and good responses the same (all 4-5) and misses
    the spans -> the critic genuinely cannot discriminate -> retrain (span head
    in particular).

This run only needs the critic (no generator), so it is a light single-GPU job.

Adv mode  ->  critic signal mapping (see report / docstring table):
  apathetic   -> empathy score        (no span type)
  assumptions -> specificity score    (no span type)
  symptoms    -> factual score        + factual span
  judgmental  -> toxicity score       + toxicity span
  medication  -> (no score dim)       + medical span
  therapy     -> (no score dim)       + medical span
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from refinement.critic import build_critic
from refinement.io import clean_text


# Adv failure mode (normalized key) -> (critic score dimension, critic span type).
# None means that channel does not exist for this mode.
MODE_TO_SIGNAL: dict[str, tuple[str | None, str | None]] = {
    "apathetic": ("empathy", None),
    "assumptions": ("specificity", None),
    "symptoms": ("factual", "factual"),
    "judgmental": ("toxicity", "toxicity"),
    "medication": (None, "medical"),
    "therapy": (None, "medical"),
}

# train_critic.py's as_float() deliberately reads the RAW CounselBench toxicity
# scale (low = safe/non-toxic, high = toxic) for every split, so the critic's
# "toxicity" head is trained with the OPPOSITE polarity from the other 4 score
# dims (overall/empathy/specificity/factual, where high = good). A "bad"
# (judgmental) response is therefore expected to score HIGHER on toxicity, not
# lower -- the discrimination test below must flip direction for this one dim
# or it reports the correct signal as if it were reversed.
REVERSED_SCORE_DIMS = {"toxicity"}

# Keyword -> normalized mode key. `mode` in the CSV is a full sentence
# ("The response is apathetic.", "The response makes assumptions...", etc.).
MODE_KEYWORDS: list[tuple[str, str]] = [
    ("apathetic", "apathetic"),
    ("assumption", "assumptions"),
    ("symptom", "symptoms"),
    ("judgment", "judgmental"),
    ("judgement", "judgmental"),
    ("judgmental", "judgmental"),
    ("medication", "medication"),
    ("therapy", "therapy"),
]

SCORE_DIMS = ["overall", "empathy", "specificity", "factual", "toxicity"]


def normalize_mode(sentence: str | None) -> str | None:
    if not sentence:
        return None
    low = sentence.lower()
    for kw, key in MODE_KEYWORDS:
        if kw in low:
            return key
    return None


def normalize_yesno(value: str | None) -> bool | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"yes", "y", "true", "1", "present"}:
        return True
    if v in {"no", "n", "false", "0", "absent"}:
        return False
    return None


def normalize_span_type(value: Any) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower()
    if "medic" in v:          # medical / medical_advice / medication
        return "medical"
    if "factual" in v or "fact" in v:
        return "factual"
    if "toxic" in v:
        return "toxicity"
    return v


def detect_columns(fieldnames: list[str]) -> dict[str, str]:
    """Best-effort map of logical fields -> actual CSV column names."""
    lower = {f.lower(): f for f in fieldnames}

    def pick(*candidates: str) -> str | None:
        for c in candidates:
            if c in lower:
                return lower[c]
        return None

    return {
        "post": pick("new_post", "post", "question_post"),
        "response": pick("new_response", "response", "answer", "model_response"),
        "mode": pick("mode", "failure_mode", "target_mode"),
        "human": pick("human_answer", "human", "label", "annotation", "answer"),
        "model": pick("model_name", "model", "responder"),
    }


def load_grouped_responses(path: Path, cols: dict[str, str]) -> list[dict[str, Any]]:
    """Group long-format rows into unique (post, response) responses.

    Each returned record carries the response text plus a {mode_key: bool}
    map of which failure modes a human marked present on that response.
    """
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            post = clean_text(row.get(cols["post"]))
            response = clean_text(row.get(cols["response"]))
            if not post or not response:
                continue
            key = (post, response)
            rec = groups.get(key)
            if rec is None:
                rec = {
                    "post": post,
                    "response": response,
                    "model_name": (row.get(cols["model"]) or "").strip() if cols["model"] else "",
                    "mode_labels": {},  # mode_key -> bool
                }
                groups[key] = rec
            mode_key = normalize_mode(row.get(cols["mode"])) if cols["mode"] else None
            human = normalize_yesno(row.get(cols["human"])) if cols["human"] else None
            if mode_key is not None and human is not None:
                # If a mode is annotated multiple times, treat "present" as sticky.
                rec["mode_labels"][mode_key] = rec["mode_labels"].get(mode_key, False) or human

    records = list(groups.values())
    for i, rec in enumerate(records):
        rec["id"] = f"adv_resp_{i:05d}"
    return records


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _auc_lower(yes: list[float], no: list[float]) -> float | None:
    """Rank-based P(score on a flagged-bad response < score on a clean one).

    1.0 = perfect separation in the expected direction (bad scored lower),
    0.5 = no discrimination, <0.5 = backwards. Ties count as 0.5.
    """
    if not yes or not no:
        return None
    wins = 0.0
    for a in yes:
        for b in no:
            if a < b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(yes) * len(no))


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)

    # --- Overall: any-failure-present vs none ---
    overall_yes, overall_no = [], []
    for r in results:
        labels = r["mode_labels"]
        any_present = any(labels.values())
        ov = r["critic_scores"].get("overall")
        if ov is None:
            continue
        (overall_yes if any_present else overall_no).append(float(ov))

    per_mode: dict[str, Any] = {}
    for mode_key, (dim, span_type) in MODE_TO_SIGNAL.items():
        yes_scores, no_scores = [], []
        det_yes_hit = det_yes_total = 0  # recall on flagged-bad
        det_no_hit = det_no_total = 0    # false-alarm rate on clean
        n_yes = n_no = 0
        for r in results:
            present = r["mode_labels"].get(mode_key)
            if present is None:
                continue
            if present:
                n_yes += 1
            else:
                n_no += 1
            # (A) score discrimination on the matching dimension
            if dim is not None:
                sc = r["critic_scores"].get(dim)
                if sc is not None:
                    (yes_scores if present else no_scores).append(float(sc))
            # (B) span detection on the matching span type
            if span_type is not None:
                flagged = span_type in r["flagged_span_types"]
                if present:
                    det_yes_total += 1
                    det_yes_hit += int(flagged)
                else:
                    det_no_total += 1
                    det_no_hit += int(flagged)

        reversed_dim = dim in REVERSED_SCORE_DIMS
        mean_yes, mean_no = _mean(yes_scores), _mean(no_scores)
        raw_auc = _auc_lower(yes_scores, no_scores)
        # For a normal dim (high=good), "bad" is expected to score LOWER, so
        # raw_auc = P(present < absent) is already the correct-direction test.
        # For a reversed dim (toxicity: high=bad), "bad" is expected to score
        # HIGHER, so the correct-direction test is P(present > absent) = 1-raw_auc.
        auc_correct_direction = (1 - raw_auc) if (reversed_dim and raw_auc is not None) else raw_auc
        delta_correct_direction = None
        if mean_yes is not None and mean_no is not None:
            delta_correct_direction = (mean_yes - mean_no) if reversed_dim else (mean_no - mean_yes)

        per_mode[mode_key] = {
            "score_dim": dim,
            "score_dim_reversed_scale": reversed_dim,
            "span_type": span_type,
            "n_present": n_yes,
            "n_absent": n_no,
            "score_mean_present": mean_yes,
            "score_mean_absent": mean_no,
            # Positive = model moves the "bad" scale value in the expected
            # direction, accounting for toxicity's reversed (high=bad) polarity.
            "score_delta_correct_direction": delta_correct_direction,
            # 0.5 = no discrimination, 1.0 = perfect, already direction-corrected.
            "score_auc_correct_direction": auc_correct_direction,
            "span_recall_on_present": (det_yes_hit / det_yes_total if det_yes_total else None),
            "span_falsealarm_on_absent": (det_no_hit / det_no_total if det_no_total else None),
        }

    return {
        "n_responses": n,
        "overall_score_mean_any_failure": _mean(overall_yes),
        "overall_score_mean_no_failure": _mean(overall_no),
        "overall_score_std_any_failure": _std(overall_yes),
        "overall_score_std_no_failure": _std(overall_no),
        "overall_auc_lower_on_failure": _auc_lower(overall_yes, overall_no),
        "n_any_failure": len(overall_yes),
        "n_no_failure": len(overall_no),
        "per_mode": per_mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CounselBench-Adv CSV with new_response + human_answer")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--critic", choices=["ours", "mock"], default="ours",
                    help="'mock' runs the no-model MockCritic for a local plumbing smoke test")
    ap.add_argument("--critic-checkpoint", default=None)
    ap.add_argument("--critic-base-model", default=None)
    ap.add_argument("--critic-code-dir", default=".")
    ap.add_argument("--critic-device-index", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="critic rationale length; scores/spans come from the heads, "
                         "so this can be small to save time")
    ap.add_argument("--limit", type=int, default=None, help="cap #responses (smoke test)")
    ap.add_argument("--dump-schema", action="store_true",
                    help="print detected columns + label distributions, then exit")
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
        sys.exit(
            f"Could not find columns for {missing}. Pass a CSV with new_post / "
            f"new_response / mode / human_answer, or edit detect_columns()."
        )

    records = load_grouped_responses(in_path, cols)
    # Report label coverage so we can sanity-check the mapping before a long run.
    mode_counts: dict[str, dict[str, int]] = {}
    for rec in records:
        for mk, present in rec["mode_labels"].items():
            d = mode_counts.setdefault(mk, {"present": 0, "absent": 0})
            d["present" if present else "absent"] += 1
    print(f"[data] {len(records)} unique (post,response) responses", flush=True)
    print(f"[data] per-mode label counts: {json.dumps(mode_counts, ensure_ascii=False)}", flush=True)

    if args.dump_schema:
        return

    if args.limit:
        records = records[: args.limit]
        print(f"[data] limited to {len(records)} responses", flush=True)

    if args.critic == "mock":
        critic = build_critic("mock")
    else:
        if not args.critic_checkpoint or not args.critic_base_model:
            sys.exit("--critic ours needs --critic-checkpoint and --critic-base-model")
        critic = build_critic(
            "ours",
            checkpoint=args.critic_checkpoint,
            code_dir=args.critic_code_dir,
            base_model=args.critic_base_model,
            device_index=args.critic_device_index,
            max_new_tokens=args.max_new_tokens,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "responses.jsonl"

    results: list[dict[str, Any]] = []
    with traces_path.open("w", encoding="utf-8", newline="\n") as handle:
        for i, rec in enumerate(records, 1):
            crit = critic.critique(rec["post"], rec["response"])
            flagged = sorted({
                t for t in (normalize_span_type(s.get("failure_type")) for s in (crit.get("spans") or []))
                if t
            })
            row = {
                "id": rec["id"],
                "model_name": rec["model_name"],
                "mode_labels": rec["mode_labels"],
                "critic_scores": crit.get("scores") or {"overall": crit.get("score")},
                "flagged_span_types": flagged,
                "n_spans": len(crit.get("spans") or []),
            }
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            ov = row["critic_scores"].get("overall")
            print(f"[{i}/{len(records)}] {rec['id']} model={rec['model_name']} "
                  f"overall={ov} spans={flagged} labels={rec['mode_labels']}", flush=True)

    summary = summarize(results)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n==== SUMMARY ====", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"\nResponses -> {traces_path}", flush=True)
    print(f"Summary   -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
