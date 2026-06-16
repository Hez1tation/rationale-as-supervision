#!/usr/bin/env python3


from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

FAILURE_DIMS = ["medical", "factual", "toxicity"]
NULL_TOKENS = {"", "n/a", "na", "null", "none", "."}


_CHAR_MAP = {
    "‘": "'", "’": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-",
    "…": ".",            # ellipsis -> single char (still 1:1)
    " ": " ",            # nbsp -> space
    "�": "?",            # replacement char (mojibake) -> placeholder
}


def normalize_keep_len(text: str) -> str:
    return "".join(_CHAR_MAP.get(ch, ch) for ch in text)


def split_snippets(copy_text: str) -> List[str]:
    """A copy cell may hold several snippets joined by ' ; '."""
    parts = [p.strip() for p in copy_text.split(" ; ")]
    return [p for p in parts if p and p.lower() not in NULL_TOKENS]


def collect_copies(raw_refer: Any) -> List[Tuple[str, str, Optional[int]]]:
    """Return (dim, copy_text, survey_id) tuples across the 3 raw_refer shapes.

    - train/dev: {dim: {"raw_text": ...}}
    - test:      [{"survey_id": int, dim: copytext, ...}, ...]
    """
    out: List[Tuple[str, str, Optional[int]]] = []
    if isinstance(raw_refer, dict):
        for dim in FAILURE_DIMS:
            cell = raw_refer.get(dim)
            if isinstance(cell, dict):
                txt = cell.get("raw_text")
            else:
                txt = cell
            if isinstance(txt, str) and txt.strip():
                out.append((dim, txt, None))
    elif isinstance(raw_refer, list):
        for entry in raw_refer:
            if not isinstance(entry, dict):
                continue
            sid = entry.get("survey_id")
            for dim in FAILURE_DIMS:
                txt = entry.get(dim)
                if isinstance(txt, str) and txt.strip():
                    out.append((dim, txt, sid))
    return out


def locate(snippet: str, response: str, fuzzy_ratio: float, min_chars: int
           ) -> Optional[Tuple[int, int, str]]:
    """Locate snippet in response -> (start, end, match_kind) or None."""
    if not snippet or not response:
        return None

    # 1) exact
    idx = response.find(snippet)
    if idx >= 0:
        return idx, idx + len(snippet), "exact"

    # 2) length-preserving normalized exact (offsets valid on original)
    nresp = normalize_keep_len(response)
    nsnip = normalize_keep_len(snippet)
    idx = nresp.find(nsnip)
    if idx >= 0:
        return idx, idx + len(nsnip), "normalized"

    # 3) fuzzy: longest contiguous matching block on normalized strings
    sm = difflib.SequenceMatcher(None, nresp, nsnip, autojunk=False)
    block = sm.find_longest_match(0, len(nresp), 0, len(nsnip))
    if block.size >= max(min_chars, int(fuzzy_ratio * len(nsnip))):
        return block.a, block.a + block.size, "fuzzy"
    return None


def merge_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge overlapping spans of the SAME failure_type."""
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for s in spans:
        by_type.setdefault(s["failure_type"], []).append(s)
    merged: List[Dict[str, Any]] = []
    for ftype, group in by_type.items():
        group.sort(key=lambda x: (x["start_char"], x["end_char"]))
        cur = None
        for s in group:
            if cur and s["start_char"] <= cur["end_char"]:
                cur["end_char"] = max(cur["end_char"], s["end_char"])
                if s["match"] == "exact":
                    cur["match"] = "exact"
            else:
                if cur:
                    merged.append(cur)
                cur = dict(s)
        if cur:
            merged.append(cur)
    return merged


def process_row(row: Dict[str, Any], args, stats: Dict[str, int]) -> Dict[str, Any]:
    response = row.get("response", "") or ""
    raw_spans: List[Dict[str, Any]] = []
    for dim, copy_text, sid in collect_copies(row.get("raw_refer")):
        for snip in split_snippets(copy_text):
            stats["pieces"] += 1
            hit = locate(snip, response, args.fuzzy_ratio, args.min_chars)
            if hit is None:
                stats["dropped"] += 1
                continue
            s, e, kind = hit
            stats[kind] += 1
            stats[f"type_{dim}"] += 1
            raw_spans.append({
                "start_char": s, "end_char": e, "failure_type": dim,
                "match": kind, "survey_id": sid,
                "text": response[s:e],
            })
    spans = merge_spans(raw_spans)
    row["span_labels"] = spans
    if spans:
        stats["rows_with_span"] += 1
    return row


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Resolve real failure spans into span_labels.")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--fuzzy_ratio", type=float, default=0.8,
                   help="min fraction of snippet length the longest match must cover")
    p.add_argument("--min_chars", type=int, default=15,
                   help="min matched chars to accept a fuzzy span")
    p.add_argument("--backup_dir", default=None, help="default: <data_dir>/orig")
    args = p.parse_args()

    backup_dir = args.backup_dir or os.path.join(args.data_dir, "orig")
    os.makedirs(backup_dir, exist_ok=True)

    grand: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        path = os.path.join(args.data_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue

        backup = os.path.join(backup_dir, f"{split}.jsonl")
        if not os.path.exists(backup):
            shutil.copy2(path, backup)

        rows = load_jsonl(path)
        stats = {k: 0 for k in
                 ["pieces", "exact", "normalized", "fuzzy", "dropped",
                  "rows_with_span", "type_medical", "type_factual", "type_toxicity"]}
        out_rows = [process_row(r, args, stats) for r in rows]

        with open(path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        stats["rows"] = len(rows)
        grand[split] = stats

    # QA report
    print("\n================ SPAN RESOLUTION REPORT ================")
    print(f"{'split':6} {'rows':>5} {'w/span':>7} {'pieces':>7} "
          f"{'exact':>6} {'norm':>5} {'fuzzy':>6} {'drop':>5} "
          f"{'med':>4} {'fac':>4} {'tox':>4} {'located%':>9}")
    for split, s in grand.items():
        loc = s["exact"] + s["normalized"] + s["fuzzy"]
        pct = (100.0 * loc / s["pieces"]) if s["pieces"] else 0.0
        print(f"{split:6} {s['rows']:>5} {s['rows_with_span']:>7} {s['pieces']:>7} "
              f"{s['exact']:>6} {s['normalized']:>5} {s['fuzzy']:>6} {s['dropped']:>5} "
              f"{s['type_medical']:>4} {s['type_factual']:>4} {s['type_toxicity']:>4} "
              f"{pct:>8.1f}%")
    print("=======================================================")
    print(f"Originals backed up under: {backup_dir}/")
    print("Each row now has `span_labels` (response-relative char offsets + failure_type).")


if __name__ == "__main__":
    main()
