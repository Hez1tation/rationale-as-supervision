"""Recompute critic_validate's summary.json from an existing responses.jsonl.

results/critic_validate.zip only contains responses.jsonl (the run's
summary.json was never saved/downloaded), and refinement/validate_critic.py's
toxicity-direction bug (see REVERSED_SCORE_DIMS there) has since been fixed.
This re-derives a corrected summary from the raw per-response dump WITHOUT
re-running the critic -- no GPU, seconds to run, safe to run locally.

Usage:
    python scripts/eval/recompute_critic_validate_summary.py \
        --responses path/to/responses.jsonl --out path/to/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from refinement.validate_critic import summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = []
    with open(args.responses, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    summary = summarize(results)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[data] {len(results)} responses loaded from {args.responses}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary -> {args.out}")


if __name__ == "__main__":
    main()
