#!/usr/bin/env python3
"""Offline re-evaluation of saved final_results.json runs. No GPU, no retraining.

What it adds over the in-training metrics:
1. Multi-reference rationale metrics: token-F1 / ROUGE-L computed against EACH
   annotator's rationale separately, taking the max per example (the in-training
   metric compares against all annotators' text concatenated into one blob,
   which punishes the model for not covering every annotator's wording).
2. Bootstrap 95% CIs for score metrics (MAE / Spearman / Kendall) and rationale
   metrics, so run-to-run differences can be judged against sampling noise
   (test split is only 80 rows).
3. Paired bootstrap comparison between two runs on the same test examples.
4. Calibration check: linear fit gold ~ a*pred + b on dev, applied to test,
   to separate "worse ordering" from "mis-scaled outputs" (MAE vs rank metrics).

Usage:
  python eval_offline.py name1=path/to/final_results.json name2=... [--data data/test.jsonl]

Token-F1 / ROUGE-L implementations mirror rationale_similarity() in
train_critic.py so numbers stay comparable with the training logs.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr

SCORE_DIMS = ["overall", "empathy", "specificity", "factual", "toxicity"]
BOOTSTRAP_B = 2000
SEED = 42


# ---------------------------------------------------------------- similarity
def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rationale_similarity(pred: str, gold: str) -> Dict[str, float]:
    p = re.findall(r"[A-Za-z0-9']+", pred.lower())
    g = re.findall(r"[A-Za-z0-9']+", gold.lower())
    if not p and not g:
        return {"token_f1": 1.0, "rougeL_f1": 1.0}
    if not p or not g:
        return {"token_f1": 0.0, "rougeL_f1": 0.0}
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    prec, rec = overlap / len(p), overlap / len(g)
    tok_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    lcs = _lcs_len(p, g)
    rprec, rrec = lcs / len(p), lcs / len(g)
    rouge_l = 2 * rprec * rrec / (rprec + rrec) if (rprec + rrec) else 0.0
    return {"token_f1": tok_f1, "rougeL_f1": rouge_l}


# ---------------------------------------------------------------- references
def annotator_references(row: Dict[str, Any]) -> List[str]:
    """One reference string per annotator: their non-null dimension rationales joined."""
    refs = []
    for ann in row.get("rationales") or []:
        if not isinstance(ann, dict):
            if isinstance(ann, str) and ann.strip():
                refs.append(ann.strip())
            continue
        parts = []
        for k, v in ann.items():
            if k == "survey_id":
                continue
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        if parts:
            # dedupe while preserving order
            seen, uniq = set(), []
            for t in parts:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            refs.append("\n".join(uniq))
    return refs


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------- statistics
def corr_metrics(g: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    if len(set(g.tolist())) < 2 or len(set(p.tolist())) < 2:
        return float("nan"), float("nan")
    s = spearmanr(g, p)[0]
    k = kendalltau(g, p)[0]
    return float(s), float(k)


def bootstrap_ci(values_fn, n: int, rng: np.random.RandomState,
                 b: int = BOOTSTRAP_B) -> Tuple[float, float, float]:
    """values_fn(idx array) -> scalar. Returns (point, lo, hi)."""
    point = values_fn(np.arange(n))
    stats = []
    for _ in range(b):
        idx = rng.randint(0, n, size=n)
        v = values_fn(idx)
        if v == v:  # not NaN
            stats.append(v)
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return "%.3f [%.3f, %.3f]" % (point, lo, hi)


# ---------------------------------------------------------------- evaluation
class RunEval:
    def __init__(self, name: str, path: str, test_rows: List[Dict[str, Any]]):
        self.name = name
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.config = data.get("config", {})
        self.preds_test = data["predictions"]["test"]
        self.preds_dev = data["predictions"].get("dev") or []
        if len(self.preds_test) != len(test_rows):
            print("WARNING: %s has %d test predictions but data has %d rows"
                  % (name, len(self.preds_test), len(test_rows)))
        # sanity: align by idx, verify response text matches the data row
        mismatches = 0
        for i, p in enumerate(self.preds_test):
            row = test_rows[i] if i < len(test_rows) else None
            if row and (p.get("response") or "").strip()[:80] != (row.get("response") or "").strip()[:80]:
                mismatches += 1
        if mismatches:
            print("WARNING: %s: %d/%d responses do not match test.jsonl order"
                  % (name, mismatches, len(self.preds_test)))

        # ---- score arrays
        self.gold = {d: np.array([float(p["gold_scores"][d]) for p in self.preds_test]) for d in SCORE_DIMS}
        self.pred = {d: np.array([float(p["pred_scores"][d]) for p in self.preds_test]) for d in SCORE_DIMS}

        # ---- rationale similarities (concat reference = legacy; multi-ref max = new)
        self.rat_concat = []
        self.rat_multimax = []
        for i, p in enumerate(self.preds_test):
            pred_r = p.get("pred_rationale") or ""
            concat = p.get("gold_rationale") or ""
            self.rat_concat.append(rationale_similarity(pred_r, concat))
            refs = annotator_references(test_rows[i]) if i < len(test_rows) else []
            if not refs:
                refs = [concat]
            sims = [rationale_similarity(pred_r, r) for r in refs]
            self.rat_multimax.append({
                "token_f1": max(s["token_f1"] for s in sims),
                "rougeL_f1": max(s["rougeL_f1"] for s in sims),
            })

    # ---- per-dimension score metrics with CI on every dim
    def score_report(self, rng: np.random.RandomState) -> Dict[str, Any]:
        out = {}
        for d in SCORE_DIMS:
            g, p = self.gold[d], self.pred[d]
            n = len(g)
            mae = bootstrap_ci(lambda idx: float(np.mean(np.abs(g[idx] - p[idx]))), n, rng)
            sp = bootstrap_ci(lambda idx: corr_metrics(g[idx], p[idx])[0], n, rng)
            kd = bootstrap_ci(lambda idx: corr_metrics(g[idx], p[idx])[1], n, rng)
            out[d] = {"mae": mae, "spearman": sp, "kendall": kd}
        return out

    def rationale_report(self, rng: np.random.RandomState) -> Dict[str, Any]:
        out = {}
        n = len(self.rat_concat)
        for label, sims in (("concat", self.rat_concat), ("multiref_max", self.rat_multimax)):
            for key in ("token_f1", "rougeL_f1"):
                arr = np.array([s[key] for s in sims])
                out["%s/%s" % (label, key)] = bootstrap_ci(lambda idx: float(np.mean(arr[idx])), n, rng)
        return out

    # ---- calibration: fit on dev (overall), apply to test
    def calibration_report(self) -> Optional[Dict[str, float]]:
        if not self.preds_dev:
            return None
        gd = np.array([float(p["gold_scores"]["overall"]) for p in self.preds_dev])
        pd_ = np.array([float(p["pred_scores"]["overall"]) for p in self.preds_dev])
        if np.std(pd_) < 1e-8:
            return None
        a, b = np.polyfit(pd_, gd, 1)
        gt, pt = self.gold["overall"], self.pred["overall"]
        cal = np.clip(a * pt + b, 1.0, 5.0)
        return {
            "fit_a": float(a), "fit_b": float(b),
            "test_mae_raw": float(np.mean(np.abs(gt - pt))),
            "test_mae_calibrated": float(np.mean(np.abs(gt - cal))),
            "pred_std": float(np.std(pt)), "gold_std": float(np.std(gt)),
            "pred_mean": float(np.mean(pt)), "gold_mean": float(np.mean(gt)),
        }


def paired_compare(run_a: RunEval, run_b: RunEval, rng: np.random.RandomState) -> Dict[str, Any]:
    """Paired bootstrap of (B - A) differences on the same test examples."""
    n = len(run_a.preds_test)
    out = {}

    def add(label, fn):
        point, lo, hi = bootstrap_ci(fn, n, rng)
        # fraction of resamples where B is NOT better, as a rough one-sided check
        out[label] = {"diff": point, "lo": lo, "hi": hi, "significant": (lo > 0) or (hi < 0)}

    ga, pa = run_a.gold["overall"], run_a.pred["overall"]
    gb, pb = run_b.gold["overall"], run_b.pred["overall"]
    add("overall_mae_diff", lambda idx: float(np.mean(np.abs(gb[idx] - pb[idx])) - np.mean(np.abs(ga[idx] - pa[idx]))))
    add("overall_spearman_diff", lambda idx: corr_metrics(gb[idx], pb[idx])[0] - corr_metrics(ga[idx], pa[idx])[0])
    fa = np.array([s["token_f1"] for s in run_a.rat_multimax])
    fb = np.array([s["token_f1"] for s in run_b.rat_multimax])
    add("rationale_multiref_token_f1_diff", lambda idx: float(np.mean(fb[idx]) - np.mean(fa[idx])))
    ra = np.array([s["rougeL_f1"] for s in run_a.rat_multimax])
    rb = np.array([s["rougeL_f1"] for s in run_b.rat_multimax])
    add("rationale_multiref_rougeL_diff", lambda idx: float(np.mean(rb[idx]) - np.mean(ra[idx])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path/to/final_results.json")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--out", default="eval_offline_results.json")
    args = ap.parse_args()

    test_rows = load_jsonl(args.data)
    ref_counts = [len(annotator_references(r)) for r in test_rows]
    print("Loaded %d test rows; annotator references per row: min=%d max=%d mean=%.1f"
          % (len(test_rows), min(ref_counts), max(ref_counts), sum(ref_counts) / len(ref_counts)))

    runs: List[RunEval] = []
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            print("Bad run spec (want name=path): %s" % spec)
            sys.exit(1)
        runs.append(RunEval(name, path, test_rows))

    results: Dict[str, Any] = {}
    for run in runs:
        rng = np.random.RandomState(SEED)  # same resamples for every run
        print("\n" + "=" * 72)
        print("RUN: %s  (%s)" % (run.name, run.path))
        print("=" * 72)

        score = run.score_report(rng)
        print("\n  Scores (test, point [95%% CI], n=%d):" % len(run.preds_test))
        print("  %-12s %-24s %-24s %-24s" % ("dim", "MAE", "Spearman", "Kendall"))
        for d in SCORE_DIMS:
            print("  %-12s %-24s %-24s %-24s" % (
                d, fmt_ci(*score[d]["mae"]), fmt_ci(*score[d]["spearman"]), fmt_ci(*score[d]["kendall"])))

        rng2 = np.random.RandomState(SEED)
        rat = run.rationale_report(rng2)
        print("\n  Rationale (test):")
        for key in ("concat/token_f1", "multiref_max/token_f1", "concat/rougeL_f1", "multiref_max/rougeL_f1"):
            print("    %-26s %s" % (key, fmt_ci(*rat[key])))

        cal = run.calibration_report()
        if cal:
            print("\n  Calibration (overall; fit on dev n=%d, applied to test):" % len(run.preds_dev))
            print("    pred mean/std = %.3f / %.3f   gold mean/std = %.3f / %.3f"
                  % (cal["pred_mean"], cal["pred_std"], cal["gold_mean"], cal["gold_std"]))
            print("    linear fit: gold ~ %.3f * pred + %.3f" % (cal["fit_a"], cal["fit_b"]))
            print("    test MAE: raw %.4f -> calibrated %.4f" % (cal["test_mae_raw"], cal["test_mae_calibrated"]))

        results[run.name] = {"scores": score, "rationale": rat, "calibration": cal,
                             "config_note": {k: run.config.get(k) for k in
                                             ("num_epochs", "span_threshold", "span_weight_cap", "span_binary")}}

    if len(runs) >= 2:
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                rng3 = np.random.RandomState(SEED)
                cmp_ = paired_compare(runs[i], runs[j], rng3)
                tag = "%s vs %s" % (runs[i].name, runs[j].name)
                print("\n" + "-" * 72)
                print("PAIRED COMPARISON (%s; positive diff = second run higher):" % tag)
                for k, v in cmp_.items():
                    flag = "  <-- significant (CI excludes 0)" if v["significant"] else ""
                    print("  %-36s %+.4f [%+.4f, %+.4f]%s" % (k, v["diff"], v["lo"], v["hi"], flag))
                results["paired/%s" % tag] = cmp_

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved detailed results to %s" % args.out)


if __name__ == "__main__":
    main()
