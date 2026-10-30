"""Input/output helpers for Stage 4 refinement runs."""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable


ADV_FAILURE_MODES = {
    "apathetic",
    "assumptions",
    "symptoms",
    "judgmental",
    "medication",
    "therapy",
}


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _adv_record(row: dict[str, Any], index: int, source: str) -> dict[str, Any] | None:
    """Map one CounselBench-Adv row (long schema) into a Stage 4 prompt record.

    The real dataset (izi-ano/CounselBench-Adv-human-annotation) is long-format:
    the adversarial client question is in `new_post`, the target failure mode is the
    sentence in `mode` (e.g. "The response is apathetic."), and `question` is an
    annotation meta-prompt (NOT the client question -- do not use it). Each post is
    repeated across `model_name`, so callers should --dedup-question.
    """
    question = clean_text(row.get("new_post") or row.get("post"))
    if not question:
        return None
    return {
        "id": f"adv_{index:05d}",
        "target_failure_mode": (row.get("mode") or "").strip(),
        "question": question,
        "metadata": {"model_name": row.get("model_name"), "source": source},
    }


def load_adv_csv(path: Path) -> list[dict[str, Any]]:
    """Load a CounselBench-Adv CSV into Stage 4 prompt records.

    Handles the real long-format schema (a `new_post` column). Falls back to the
    legacy wide format (one column per failure mode) if `new_post` is absent.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "new_post" in fieldnames:
            records = [_adv_record(row, i, path.name) for i, row in enumerate(reader)]
            return [r for r in records if r is not None]

        # Legacy wide format: one column per failure mode, each holding a question.
        rows: list[dict[str, Any]] = []
        for row_idx, row in enumerate(reader):
            for mode, question in row.items():
                if mode not in ADV_FAILURE_MODES:
                    continue
                question = clean_text(question)
                if not question:
                    continue
                rows.append(
                    {
                        "id": f"adv_{mode}_{row_idx:03d}",
                        "target_failure_mode": mode,
                        "question": question,
                        "source_file": path.name,
                    }
                )
        return rows


def load_hf_adv(repo_id: str, split: str = "train") -> list[dict[str, Any]]:
    """Load a CounselBench-Adv dataset straight from the HuggingFace Hub.

    Requires the `datasets` library. Use via `--input hf:<repo_id>`, e.g.
    `--input hf:izi-ano/CounselBench-Adv-human-annotation`.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "Loading hf: datasets needs the `datasets` library. "
            "Install it in the env: pip install datasets"
        ) from exc

    dataset = load_dataset(repo_id, split=split)
    records = [_adv_record(dict(ex), i, repo_id) for i, ex in enumerate(dataset)]
    return [r for r in records if r is not None]


def normalize_prompt_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = clean_text(row.get("question") or row.get("prompt") or row.get("new_post"))
    if not question:
        raise ValueError(f"Record {index} is missing a question/prompt field.")
    return {
        "id": row.get("id") or row.get("question_id") or f"record_{index:05d}",
        "target_failure_mode": row.get("target_failure_mode") or row.get("mode"),
        "question": question,
        "metadata": {k: v for k, v in row.items() if k not in {"question", "prompt", "new_post"}},
    }


def load_prompt_records(source: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    source_str = str(source)
    if source_str.startswith("hf:"):
        rows = load_hf_adv(source_str[3:])
    else:
        path = Path(source)
        if path.suffix.lower() == ".csv":
            rows = load_adv_csv(path)
        else:
            rows = [normalize_prompt_record(row, i) for i, row in enumerate(load_jsonl(path))]
    if limit is not None:
        rows = rows[:limit]
    return rows


def summarize_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stop_counts: dict[str, int] = {}
    initial_scores: list[float] = []
    scores: list[float] = []
    iterations: list[int] = []
    num_refined = 0
    quality_gate_record_count = 0
    quality_gate_event_count = 0
    artifact_resolved_count = 0
    final_artifact_count = 0
    final_artifact_record_count = 0
    final_clean_count = 0
    for row in rows:
        stop_reason = row.get("stop_reason", "unknown")
        stop_counts[stop_reason] = stop_counts.get(stop_reason, 0) + 1
        row_iterations = row.get("iterations") or []
        refined = any(not iteration.get("stopped", False) for iteration in row_iterations)
        if refined:
            num_refined += 1

        quality_gate_triggered = False
        for iteration in row_iterations:
            critique = iteration.get("critique") or {}
            quality_gate = critique.get("quality_gate") or {}
            if quality_gate.get("triggered"):
                quality_gate_triggered = True
                quality_gate_event_count += 1
        if quality_gate_triggered:
            quality_gate_record_count += 1

        if row_iterations:
            initial_critique = row_iterations[0].get("critique") or {}
            if initial_critique.get("score") is not None:
                initial_scores.append(float(initial_critique["score"]))
        if row.get("final_score") is not None:
            scores.append(float(row["final_score"]))
        iterations.append(len(row_iterations))

        final_validation = row.get("final_validation") or {}
        if final_validation.get("is_clean"):
            final_clean_count += 1
            if quality_gate_triggered:
                artifact_resolved_count += 1
        artifact_count = int(final_validation.get("artifact_count") or 0)
        if artifact_count:
            final_artifact_record_count += 1
        final_artifact_count += artifact_count

    mean_initial_score = sum(initial_scores) / len(initial_scores) if initial_scores else None
    mean_final_score = sum(scores) / len(scores) if scores else None

    return {
        "num_records": len(rows),
        "stop_counts": stop_counts,
        "mean_initial_score": mean_initial_score,
        "mean_final_score": mean_final_score,
        "mean_score_delta": (
            mean_final_score - mean_initial_score
            if mean_initial_score is not None and mean_final_score is not None
            else None
        ),
        "mean_iterations": sum(iterations) / len(iterations) if iterations else None,
        "num_refined": num_refined,
        "refinement_rate": num_refined / len(rows) if rows else None,
        "quality_gate_record_count": quality_gate_record_count,
        "quality_gate_event_count": quality_gate_event_count,
        "artifact_resolved_count": artifact_resolved_count,
        "artifact_resolved_rate": (
            artifact_resolved_count / quality_gate_record_count
            if quality_gate_record_count
            else None
        ),
        "final_artifact_count": final_artifact_count,
        "final_artifact_record_count": final_artifact_record_count,
        "final_artifact_rate": final_artifact_record_count / len(rows) if rows else None,
        "final_clean_count": final_clean_count,
        "final_clean_rate": final_clean_count / len(rows) if rows else None,
    }
