#!/usr/bin/env python3
"""Dump a fixed random sample of test examples for manual/LLM judging:
gold annotator rationales vs predicted rationale from each run."""
import json
import random

RUNS = {
    "softmax": "final_results.json",
    "sigmoid": "out_base_sigmoid/final_results.json",
}
N_SAMPLE = 20
SEED = 42


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def trunc(s, n):
    s = (s or "").replace("\r", " ").strip()
    return s if len(s) <= n else s[:n] + " ...[truncated]"


test_rows = load_jsonl("data/test.jsonl")
preds = {}
for name, path in RUNS.items():
    with open(path, "r", encoding="utf-8") as f:
        preds[name] = json.load(f)["predictions"]["test"]

idxs = sorted(random.Random(SEED).sample(range(len(test_rows)), N_SAMPLE))

with open("judge_sample.txt", "w", encoding="utf-8") as out:
    for i in idxs:
        row = test_rows[i]
        out.write("#" * 80 + "\n")
        out.write("EXAMPLE %d | gold overall=%s\n" % (i, preds["sigmoid"][i]["gold_scores"]["overall"]))
        out.write("QUESTION: %s\n" % trunc(row["question"], 250))
        out.write("RESPONSE: %s\n" % trunc(row["response"], 600))
        out.write("--- GOLD (per annotator) ---\n")
        for ann in row.get("rationales") or []:
            parts = ["%s: %s" % (k, v.strip()) for k, v in ann.items()
                     if k != "survey_id" and isinstance(v, str) and v.strip()]
            if parts:
                out.write("  * " + " | ".join(parts) + "\n")
        for name in RUNS:
            out.write("--- PRED [%s] (pred overall=%.2f) ---\n"
                      % (name, preds[name][i]["pred_scores"]["overall"]))
            out.write("  %s\n" % trunc(preds[name][i].get("pred_rationale"), 700))
        out.write("\n")
print("Wrote judge_sample.txt with %d examples: %s" % (len(idxs), idxs))
