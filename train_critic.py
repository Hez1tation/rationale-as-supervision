#!/usr/bin/env python3
"""Train a rationale critic on Qwen2.5-7B-Instruct.

Compared with ``train_critic_llama3_1.py`` (score + span heads only, AutoModel
backbone, no text generation), this script keeps the same data layout / schema
handling but switches to a CausalLM backbone so the critic can ALSO generate a
natural-language rationale (the "critique generation head" from the project
schedule, Stage 3).

One backbone (Qwen2.5-7B-Instruct + LoRA), three supervision signals:

    L_total = w_score * L_score        (MSE over 5 numeric score fields)
            + w_span  * L_span         (token-level multi-label BCE, one sigmoid
                                        per failure type, per-type pos_weight)
            + w_rat   * L_rationale    (causal-LM loss over the gold rationale)

Notes
-----
* Span supervision uses the REAL failure-mode spans resolved by
  ``prepare_spans.py`` (from CounselBench ``*_copy`` annotations) and stored in
  each row's ``span_labels`` field. If a row lacks ``span_labels`` the script
  falls back to weak lexical-overlap pseudo spans (with a warning).
* The span head is MULTI-LABEL: ~25% of annotated failure chars carry more than
  one type (e.g. the same sentence flagged as both factual and toxicity), so a
  single-label softmax would get contradictory supervision. Each failure type
  gets its own sigmoid; overlapping gold spans light up several types at once.
  Span metrics are micro-aggregated (pooled TP/FP/FN over all tokens of all
  examples), so examples where the model predicts nothing still count.
* The user asked for "loss(span + score)". Those two are logged separately and
  prominently. ``L_rationale`` is required for the generation head to learn, so
  it is part of the total loss too; set ``--rationale_loss_weight 0`` to disable
  generation training (``pred_rationale`` then becomes meaningless).
* Generation target (``--target_format structured``, the default) is a
  span-grounded critique instead of the old free-form concat::

      ### Problems:
      - [factual] "<span quoted from the response>": <annotator's dim rationale>
      - [toxicity] (not localized): <rationale whose span couldn't be located>
      ### Overall assessment:
      <cleaned, deduped overall rationales>
      ### Overall: 3/5

  Targets are cleaned: "na"/"Nil" filler dropped, cross-response comparisons
  ("not as thorough as responses 1 and 2") removed sentence-wise, and
  "no problem here" dimension statements filtered out of Problems. Quotes come
  ONLY from prepare_spans.py-located spans (raw_refer copies that failed to
  locate are often annotator meta-commentary). ``--target_format plain``
  restores the legacy "### Rationale:" concat target.

Outputs
-------
* ``train.out``                  : full process log (also goes to stdout).
* ``<output_dir>/final_results.json`` : a SINGLE json file containing the run
  config, aggregate metrics, and per-example records with
  ``pred_scores`` vs ``gold_scores`` and ``pred_rationale`` vs ``gold_rationale``.
* ``<output_dir>/best`` , ``<output_dir>/final`` : LoRA + heads checkpoints.

Train schema (per line):
{
  "question": "...", "response": "...",
  "scores": {"overall": {"raw": 2, "normalized": 2}, "empathy": {...}, ...,
             "medical": {"label": "No"}, "factual": {"raw": "4", ...}, ...},
  "rationales": {"overall": "...", "medical": null, ...},
  "metadata": {"question_id": "...", "topic": "..."}
}
Dev / Test schema (per line):
{
  "question": "...", "response": "...",
  "scores": {"overall": 1, "empathy": 2, ..., "medical": "No"},
  "rationales": [{"survey_id": 28, "overall": "...", ...}, ...],
  "metadata": {"question_id": "...", "topic": "..."}
}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from contextlib import nullcontext as _nullcontext
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import kendalltau, spearmanr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger("qwen_critic")


def setup_logging(log_file: str) -> None:
    """Log to both stdout and ``log_file`` (default: train.out)."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        ensure_dir(os.path.dirname(os.path.abspath(log_file)) or ".")
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

NUMERIC_SCORE_FIELDS = ["overall", "empathy", "specificity", "factual", "toxicity"]
PAD_LABEL = -100

# Span head label scheme: one independent sigmoid per failure type (multi-label).
# There is NO "none" class — none = all sigmoids off.
SPAN_FAILURE_DIMS = ["medical", "factual", "toxicity"]


def span_label_scheme(typed: bool) -> Tuple[List[str], Dict[str, int]]:
    """Return (failure_class_names, dim_to_idx).

    typed=True : ["medical", "factual", "toxicity"]  (3 sigmoid heads)
    typed=False: ["problem"]                         (1 sigmoid head)
    """
    classes = list(SPAN_FAILURE_DIMS) if typed else ["problem"]
    return classes, {c: i for i, c in enumerate(classes)}


def failure_type_to_indices(ftype: str, dim_to_idx: Dict[str, int]) -> List[int]:
    """Map a failure type to the sigmoid head indices it lights up.

    An unknown type (only the legacy pseudo-span fallback produces one) is
    treated as type-agnostic and lights up every head.
    """
    idx = dim_to_idx.get(as_str(ftype).lower())
    return [idx] if idx is not None else list(range(len(dim_to_idx)))

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "at", "by", "from", "as", "you", "your", "i", "we",
    "they", "he", "she", "them", "his", "her", "our", "their", "my", "me",
    "am", "do", "does", "did", "not", "no", "but", "if", "so", "too",
    "about", "into", "over", "under", "again", "further", "then", "once",
    "can", "could", "should", "would", "may", "might", "will", "shall", "must",
}


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_init_distributed() -> Tuple[bool, int, int, int]:
    """If launched via torchrun (WORLD_SIZE>1), init NCCL process group.

    Returns (is_distributed, rank, world_size, local_rank). Single-GPU runs
    return (False, 0, 1, 0) and behave exactly as before.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # Large timeout: at the end of training, rank 0 alone runs generation-based
        # eval over dev+test (can take tens of minutes) while other ranks wait at a
        # barrier. The default 10-min NCCL timeout would abort them; 4h is safe.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    if isinstance(value, dict):
        # Prefer RAW: the cleaning step reverse-coded toxicity into `normalized`
        # (= 6 - raw) for train/dev only, while test stores the flat raw value.
        # Reading `raw` everywhere keeps all splits on the same (source) scale.
        for key in ("raw", "normalized", "value"):
            if key in value:
                return as_float(value.get(key))
    return None


def as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def unique_join(parts: Sequence[str], sep: str = "\n") -> str:
    seen, kept = set(), []
    for p in parts:
        txt = as_str(p)
        if not txt or txt in seen:
            continue
        seen.add(txt)
        kept.append(txt)
    return sep.join(kept).strip()


def flatten_strings(obj: Any, skip_keys: Optional[set] = None) -> List[str]:
    skip_keys = skip_keys or set()
    out: List[str] = []

    def rec(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
        elif isinstance(x, dict):
            for k, v in x.items():
                if k in skip_keys:
                    continue
                rec(v)
        elif isinstance(x, list):
            for item in x:
                rec(item)

    rec(obj)
    return out


def extract_question_id(ex: Dict[str, Any]) -> str:
    meta = ex.get("metadata", {}) or {}
    if isinstance(meta, dict):
        return as_str(meta.get("question_id") or meta.get("id") or meta.get("qid"))
    return ""


def extract_topic(ex: Dict[str, Any]) -> str:
    meta = ex.get("metadata", {}) or {}
    if isinstance(meta, dict):
        return as_str(meta.get("topic") or meta.get("title"))
    return ""


def extract_gold_scores(ex: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Unify train (nested dict) and dev/test (flat) score schemas."""
    scores = ex.get("scores", {}) or {}
    out: Dict[str, Optional[float]] = {}
    for field in NUMERIC_SCORE_FIELDS:
        out[field] = as_float(scores.get(field))
    return out


def extract_gold_rationale(ex: Dict[str, Any]) -> str:
    """Concatenate all available rationale texts into one gold rationale.

    Handles both train (dict) and dev/test (list of per-survey dicts) schemas.
    ``survey_id`` keys are skipped.
    """
    rats = ex.get("rationales", None)
    pieces = flatten_strings(rats, skip_keys={"survey_id"})
    text = unique_join(pieces, sep="\n")
    return text


# Annotator filler that must not become a generation target ("na", "Nil", ...).
_FILLER_RE = re.compile(r"^(?:n/?\.?a\.?|nil\.?|none\.?|no\.?|-+|x+|\.+)$", re.I)
# Annotators sometimes compare against sibling responses ("not as thorough as
# responses 1 and 2") that the model can never see -> drop those sentences,
# otherwise the critic learns to hallucinate cross-response comparisons.
_CROSS_REF_RE = re.compile(
    r"respons\w*\s*(?:#?\s*\d|one|two|three|four)\b"
    r"|(?:first|second|third|fourth|1st|2nd|3rd|4th|last|previous|prior|other)\s+respons"
    r"|(?:all|both)\s+(?:of\s+)?(?:the\s+)?(?:other\s+)?responses",
    re.I)


def clean_rationale_text(text: Any) -> str:
    """Clean one annotator rationale for use as a generation target."""
    t = as_str(text).strip()
    if len(t) <= 2 or _FILLER_RE.match(t):
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", t)
    kept = [s for s in sentences if s.strip() and not _CROSS_REF_RE.search(s)]
    return " ".join(kept).strip()


def extract_overall_rationale(ex: Dict[str, Any]) -> str:
    """Only the ``overall`` rationale texts (cleaned, deduped), no dim texts."""
    rats = ex.get("rationales")
    pieces: List[str] = []
    if isinstance(rats, dict):
        pieces.append(as_str(rats.get("overall")))
    elif isinstance(rats, list):
        for ann in rats:
            if isinstance(ann, dict):
                pieces.append(as_str(ann.get("overall")))
    cleaned = [c for c in (clean_rationale_text(p) for p in pieces) if c]
    return unique_join(cleaned, sep="\n")


def _clean_quote(text: Any, max_chars: int = 160) -> str:
    # inner double quotes would break the '- [dim] "...":' line format
    q = re.sub(r"\s+", " ", as_str(text)).strip().replace('"', "'")
    if len(q) > max_chars:
        q = q[:max_chars].rstrip() + "..."
    return q


# Annotators often fill a dimension's rationale with a "no problem here"
# statement ("There's nothing toxic.", "factually consistent") — those must not
# become Problems items. The adversative guard keeps mixed statements that do
# carry a criticism ("not toxic, just not very helpful overall").
_NO_ISSUE_RE = re.compile(
    r"\b(?:no|nothing|none|isn't|not)\b[^.]*?"
    r"\b(?:issue|problem|toxic|incorrect|inaccur|wrong|harm|concern|medical advice|misinformation|error)"
    r"|\bfactually\s+(?:consistent|correct|accurate)\b"
    r"|\b(?:don'?t|do not|didn'?t|can'?t|cannot)\s+see\s+anything\b"
    r"|\b(?:is|was|seems|appears|sounds)\s+(?:accurate|correct|fine|appropriate|helpful|respectful|kind|supportive|good)\b",
    re.I)
_ADVERSATIVE_RE = re.compile(r"\b(?:but|however|although|though|just|still|yet|except)\b", re.I)


def is_no_issue_statement(rationale: str) -> bool:
    return bool(_NO_ISSUE_RE.search(rationale)
                and not _ADVERSATIVE_RE.search(rationale)
                and len(rationale) <= 140)


def extract_problem_items(ex: Dict[str, Any]) -> List[Dict[str, str]]:
    """Dimension-level problems: [{dim, rationale, quote}].

    Each item pairs an annotator's medical/factual/toxicity rationale with the
    response text they copied (``span_labels`` text when located by
    prepare_spans.py, ``raw_refer`` fallback otherwise; "" when nothing was
    copied). Handles the per-annotator dict schema (train/dev) and the
    per-survey list schema (test).
    """
    rats = ex.get("rationales")
    span_labels = ex.get("span_labels") or []
    raw_refer = ex.get("raw_refer")
    items: List[Dict[str, str]] = []

    def located_quote(dim: str, survey_id: Optional[int]) -> str:
        for s in span_labels:
            if as_str(s.get("failure_type")).lower() != dim:
                continue
            sid = s.get("survey_id")
            if survey_id is not None and sid is not None and sid != survey_id:
                continue
            t = as_str(s.get("text")).strip()
            if t:
                return t
        return ""

    def add(dim: str, text: Any, survey_id: Optional[int]) -> None:
        rationale = clean_rationale_text(text)
        if not rationale or is_no_issue_statement(rationale):
            return
        # Quote ONLY spans that prepare_spans.py located in the response.
        # Unlocated raw_refer copies are frequently annotator meta-commentary
        # ("All of it", "See my response to Response 1") — quoting those would
        # teach the model to fabricate quotes that aren't in the response.
        quote = located_quote(dim, survey_id)
        items.append({"dim": dim, "rationale": rationale,
                      "quote": _clean_quote(quote) if quote else ""})

    if isinstance(rats, dict):
        for dim in SPAN_FAILURE_DIMS:
            add(dim, rats.get(dim), None)
    elif isinstance(rats, list):
        for ann in rats:
            if isinstance(ann, dict):
                for dim in SPAN_FAILURE_DIMS:
                    add(dim, ann.get(dim), ann.get("survey_id"))

    seen, uniq = set(), []
    for it in items:
        key = (it["dim"], it["rationale"].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return uniq


def extract_gold_typed_spans(ex: Dict[str, Any]) -> Optional[List[Tuple[int, int, str]]]:
    """Real, response-relative failure spans produced by ``prepare_spans.py``.

    Returns a list of (start_char, end_char, failure_type) tuples, or ``None``
    when the row was never pre-processed (caller falls back to pseudo spans).
    """
    sl = ex.get("span_labels")
    if sl is None:
        return None
    spans: List[Tuple[int, int, str]] = []
    for s in sl:
        try:
            spans.append((int(s["start_char"]), int(s["end_char"]),
                          as_str(s.get("failure_type", "problem")).lower()))
        except (KeyError, TypeError, ValueError):
            continue
    return spans


_PSEUDO_SPAN_WARNED = False


def resolve_typed_response_spans(ex: Dict[str, Any], response: str,
                                 gold_rationale: str) -> List[Tuple[int, int, str]]:
    """Prefer real typed spans from ``span_labels``; fall back to pseudo spans.

    Pseudo spans carry no failure type, so they are labelled "problem"
    (type-agnostic: ``failure_type_to_indices`` lights up every head).
    """
    global _PSEUDO_SPAN_WARNED
    real = extract_gold_typed_spans(ex)
    if real is not None:
        return real
    if not _PSEUDO_SPAN_WARNED:
        logger.warning("No `span_labels` found — falling back to weak lexical-overlap "
                       "pseudo spans. Run prepare_spans.py to use real copy annotations.")
        _PSEUDO_SPAN_WARNED = True
    return [(s, e, "problem") for (s, e) in select_pseudo_response_spans(response, gold_rationale)]


# -----------------------------------------------------------------------------
# Prompt + target construction
# -----------------------------------------------------------------------------

NULL_RATIONALE = "No issues identified on the evaluated dimensions."
NO_PROBLEMS_LINE = "- None."
TASK_LINE_PLAIN = (
    "Identify the problems in the response, write a clinical rationale, "
    "then give an overall score from 1 to 5."
)
TASK_LINE_STRUCTURED = (
    "List the problems in the response: for each one, tag it as [medical], "
    "[factual], or [toxicity], quote the problematic part of the response, and "
    "explain why it is a problem (write \"- None.\" if there are none). Then "
    "write an overall clinical assessment and give an overall score from 1 to 5."
)

# Set from --target_format in train(); module-level so build_prompt/build_target
# and model.predict stay in sync without threading the flag everywhere.
ACTIVE_TARGET_FORMAT = "structured"


def build_prompt(question: str, response: str) -> Tuple[str, int, int]:
    """Return (prompt_text, response_start_char, response_end_char)."""
    task = TASK_LINE_STRUCTURED if ACTIVE_TARGET_FORMAT == "structured" else TASK_LINE_PLAIN
    prefix = f"### Question:\n{question.strip()}\n\n### Response:\n"
    resp = response.strip()
    prompt = prefix + resp + f"\n\n### Task:\n{task}\n\n"
    return prompt, len(prefix), len(prefix) + len(resp)


def build_target(ex: Dict[str, Any], gold_rationale: str,
                 gold_overall: Optional[float]) -> str:
    overall = int(round(gold_overall)) if gold_overall is not None else 3
    overall = min(5, max(1, overall))
    if ACTIVE_TARGET_FORMAT != "structured":
        rationale = gold_rationale.strip() or NULL_RATIONALE
        return f"### Rationale:\n{rationale}\n\n### Overall: {overall}/5"

    problems = extract_problem_items(ex)
    lines = []
    for it in problems:
        quote = f'"{it["quote"]}"' if it["quote"] else "(not localized)"
        lines.append(f"- [{it['dim']}] {quote}: {it['rationale']}")
    problems_block = "\n".join(lines) if lines else NO_PROBLEMS_LINE
    assessment = extract_overall_rationale(ex)
    if not assessment:
        assessment = "See the problems above." if problems else NULL_RATIONALE
    return (f"### Problems:\n{problems_block}\n\n"
            f"### Overall assessment:\n{assessment}\n\n"
            f"### Overall: {overall}/5")


_PROBLEM_LINE_RE = re.compile(
    r"^-\s*\[(\w+)\]\s*(?:\"(.*?)\"|\((?:not localized|unlocalized)\))?\s*:?\s*(.*)$")


def parse_generated_critique(gen_text: str) -> Tuple[str, List[Dict[str, str]], str]:
    """Parse generated text into (rationale, problems, assessment).

    Structured format: problems come from the "### Problems:" block, rationale
    is the "### Overall assessment:" section. Plain format (or unparseable
    output): rationale falls back to the legacy "### Rationale:" extraction and
    problems stay empty.
    """
    problems: List[Dict[str, str]] = []
    mp = re.search(r"###\s*Problems:\s*(.*?)(?=###\s*Overall|$)", gen_text, re.S | re.I)
    if mp:
        for line in mp.group(1).strip().splitlines():
            lm = _PROBLEM_LINE_RE.match(line.strip())
            if lm and lm.group(1).lower() != "none":
                problems.append({"dim": lm.group(1).lower(),
                                 "quote": (lm.group(2) or "").strip(),
                                 "rationale": lm.group(3).strip()})
    ma = re.search(r"###\s*Overall assessment:\s*(.*?)(?=###\s*Overall\s*:|$)",
                   gen_text, re.S | re.I)
    if ma:
        return ma.group(1).strip(), problems, ma.group(1).strip()
    if mp:
        return mp.group(1).strip(), problems, ""
    m = re.search(r"###\s*Rationale:\s*(.*?)(?:###\s*Overall|$)", gen_text, re.S | re.I)
    if m:
        return m.group(1).strip(), problems, ""
    return gen_text, problems, ""


# -----------------------------------------------------------------------------
# Weak pseudo span labels (no gold spans in the data)
# -----------------------------------------------------------------------------

def normalize_for_match(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9']+", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def split_response_sentences_with_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        sent = m.group().strip()
        if sent:
            spans.append((m.start(), m.end(), sent))
    if not spans and text.strip():
        spans.append((0, len(text), text.strip()))
    return spans


def merge_char_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        last = merged[-1]
        if s <= last[1]:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def select_pseudo_response_spans(response: str, rationale: str, max_spans: int = 2) -> List[Tuple[int, int]]:
    rationale_terms = set(normalize_for_match(rationale))
    if not rationale_terms:
        return []
    scored: List[Tuple[float, int, int]] = []
    for s, e, sent in split_response_sentences_with_spans(response):
        sent_terms = set(normalize_for_match(sent))
        if not sent_terms:
            continue
        overlap = len(sent_terms & rationale_terms)
        if overlap == 0:
            continue
        jacc = overlap / max(1, len(sent_terms | rationale_terms))
        scored.append((jacc + 0.05 * overlap, s, e))
    if not scored:
        return []
    scored.sort(reverse=True)
    best = scored[0][0]
    selected: List[Tuple[int, int]] = []
    for score, s, e in scored:
        if len(selected) >= max_spans:
            break
        if score >= max(0.08, best * 0.70):
            selected.append((s, e))
    if not selected:
        selected.append((scored[0][1], scored[0][2]))
    return merge_char_spans(selected)


def typed_char_spans_to_multihot(
    offsets: Sequence[Tuple[int, int]],
    response_start: int,
    response_end: int,
    typed_spans: List[Tuple[int, int, str]],
    dim_to_idx: Dict[str, int],
) -> List[List[float]]:
    """Per-token multi-hot failure labels, shape [T, K].

    Overlapping gold spans of different types light up several heads on the
    same token (no override). Non-response tokens stay all-zero; they are
    excluded from the loss by ``response_mask``.
    """
    K = len(dim_to_idx)
    multihot = [[0.0] * K for _ in offsets]
    for cs, ce, ftype in typed_spans:
        heads = failure_type_to_indices(ftype, dim_to_idx)
        a, b = response_start + cs, response_start + ce
        for i, (s, e) in enumerate(offsets):
            if e <= s or s < response_start or e > response_end:
                continue
            if not (e <= a or s >= b):  # token overlaps span
                for k in heads:
                    multihot[i][k] = 1.0
    return multihot


def decode_multihot_spans(
    offsets: Sequence[Tuple[int, int]],
    multihot: Sequence[Sequence[float]],
    response: str,
    response_start: int,
    response_end: int,
    classes: List[str],
) -> List[Dict[str, Any]]:
    """Decode contiguous fired runs of EACH class into typed spans.

    Classes are decoded independently, so one token can contribute to several
    overlapping spans of different types.
    """
    spans: List[Dict[str, Any]] = []
    for k, name in enumerate(classes):
        cur_s: Optional[int] = None
        cur_e: Optional[int] = None

        def flush() -> None:
            nonlocal cur_s, cur_e
            if cur_s is not None and cur_e is not None and cur_e > cur_s:
                rel_s = max(0, cur_s - response_start)
                rel_e = max(0, cur_e - response_start)
                spans.append({"start_char": rel_s, "end_char": rel_e,
                              "text": response[rel_s:rel_e], "failure_type": name})
            cur_s = cur_e = None

        for (s, e), row in zip(offsets, multihot):
            fired = row[k] > 0 and e > s and response_start <= s and e <= response_end
            if not fired:
                flush()
                continue
            if cur_s is not None and s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                flush()
                cur_s, cur_e = s, e
        flush()
    spans.sort(key=lambda d: (d["start_char"], d["end_char"]))
    return spans


def new_span_counts(num_types: int) -> Dict[str, Any]:
    """Zeroed micro-aggregation counters for span metrics."""
    return {"d_tp": 0, "d_fp": 0, "d_fn": 0,
            "t_tp": [0] * num_types, "t_fp": [0] * num_types, "t_fn": [0] * num_types}


def accumulate_span_counts(tot: Dict[str, Any],
                           gold_mh: Sequence[Sequence[float]],
                           pred_mh: Sequence[Sequence[float]],
                           valid: Sequence[bool]) -> None:
    """Pool token-level TP/FP/FN into ``tot`` (micro aggregation).

    - detect (d_*): any-type-fired vs any-type-gold, type-agnostic.
    - typed  (t_*): per (token, type) pair, one counter set per type.
    """
    K = len(tot["t_tp"])
    for g, p, v in zip(gold_mh, pred_mh, valid):
        if not v:
            continue
        gpos, ppos = any(x > 0 for x in g), any(x > 0 for x in p)
        if gpos and ppos:
            tot["d_tp"] += 1
        elif ppos:
            tot["d_fp"] += 1
        elif gpos:
            tot["d_fn"] += 1
        for k in range(K):
            gk, pk = g[k] > 0, p[k] > 0
            if gk and pk:
                tot["t_tp"][k] += 1
            elif pk:
                tot["t_fp"][k] += 1
            elif gk:
                tot["t_fn"][k] += 1


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    pr = tp / (tp + fp) if (tp + fp) else float("nan")
    rc = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return pr, rc, f1


def span_summary_from_counts(tot: Dict[str, Any], classes: List[str]) -> Dict[str, Any]:
    dp, dr, df = prf_from_counts(tot["d_tp"], tot["d_fp"], tot["d_fn"])
    # typed micro: pool (token, type) pairs across all types
    mtp, mfp, mfn = sum(tot["t_tp"]), sum(tot["t_fp"]), sum(tot["t_fn"])
    tp_, tr_, tf_ = prf_from_counts(mtp, mfp, mfn)
    per_type = {}
    for k, name in enumerate(classes):
        p, r, f = prf_from_counts(tot["t_tp"][k], tot["t_fp"][k], tot["t_fn"][k])
        per_type[name] = {"precision": p, "recall": r, "f1": f,
                          "support_tokens": tot["t_tp"][k] + tot["t_fn"][k]}
    return {"detect_precision": dp, "detect_recall": dr, "detect_f1": df,
            "typed_precision": tp_, "typed_recall": tr_, "typed_f1": tf_,
            "per_type": per_type}


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_numeric_metrics(golds: Sequence[float], preds: Sequence[float]) -> Dict[str, float]:
    result = {"mae": float("nan"), "spearman": float("nan"), "kendall": float("nan")}
    if not golds:
        return result
    g = list(map(float, golds))
    p = list(map(float, preds))
    result["mae"] = sum(abs(a - b) for a, b in zip(g, p)) / len(g)
    if len(set(g)) > 1 and len(set(p)) > 1:
        result["spearman"] = float(spearmanr(g, p).statistic)
        result["kendall"] = float(kendalltau(g, p).statistic)
    return result


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
    """Lightweight, dependency-free rationale comparison: token-F1 + ROUGE-L."""
    p = re.findall(r"[A-Za-z0-9']+", pred.lower())
    g = re.findall(r"[A-Za-z0-9']+", gold.lower())
    if not p and not g:
        return {"token_f1": 1.0, "rougeL_f1": 1.0}
    if not p or not g:
        return {"token_f1": 0.0, "rougeL_f1": 0.0}
    # token-level F1 (multiset overlap)
    from collections import Counter
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    prec = overlap / len(p)
    rec = overlap / len(g)
    tok_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # ROUGE-L (LCS-based F1)
    lcs = _lcs_len(p, g)
    rprec = lcs / len(p)
    rrec = lcs / len(g)
    rouge_l = 2 * rprec * rrec / (rprec + rrec) if (rprec + rrec) else 0.0
    return {"token_f1": tok_f1, "rougeL_f1": rouge_l}


def average_key(dicts: List[Dict[str, float]], key: str) -> float:
    vals = [d[key] for d in dicts if key in d and d[key] == d[key]]
    return sum(vals) / len(vals) if vals else float("nan")


def compute_span_pos_weights(examples: List[Dict[str, Any]], dim_to_idx: Dict[str, int],
                             cap: float) -> List[float]:
    """Per-type BCE pos_weight = neg_chars / pos_chars, capped at ``cap``.

    Each sigmoid head fights its own imbalance: positives of a type cover only
    ~1-4% of response chars, so without pos_weight every head would learn to
    stay silent. Estimated from char coverage of ``span_labels`` (fast, no
    tokenization); overlapping same-type spans are de-duplicated via char sets.
    cap=1.0 disables weighting (uniform BCE).
    """
    K = len(dim_to_idx)
    pos = [0.0] * K
    total = 0.0
    for ex in examples:
        resp = as_str(ex.get("response", ""))
        if not resp:
            continue
        total += len(resp)
        per_type_chars: List[set] = [set() for _ in range(K)]
        for s in (ex.get("span_labels") or []):
            try:
                a, b = int(s["start_char"]), int(s["end_char"])
            except (KeyError, TypeError, ValueError):
                continue
            for k in failure_type_to_indices(s.get("failure_type", "problem"), dim_to_idx):
                per_type_chars[k].update(range(max(0, a), min(len(resp), b)))
        for k in range(K):
            pos[k] += len(per_type_chars[k])
    return [min(cap, (total - pos[k]) / pos[k]) if pos[k] > 0 else cap for k in range(K)]


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class CriticDataset(Dataset):
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int,
                 dim_to_id: Dict[str, int], typed: bool):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dim_to_id = dim_to_id
        self.typed = typed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        question = as_str(ex.get("question", ""))
        response = as_str(ex.get("response", ""))
        gold_scores = extract_gold_scores(ex)
        gold_rationale = extract_gold_rationale(ex)

        prompt, resp_start, resp_end = build_prompt(question, response)
        target = build_target(ex, gold_rationale, gold_scores.get("overall"))
        full_text = prompt + target

        enc = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets = enc["offset_mapping"]
        prompt_len_chars = len(prompt)

        # span labels + response mask live inside the response region of the prompt
        typed_spans = resolve_typed_response_spans(ex, response, gold_rationale)
        span_labels = typed_char_spans_to_multihot(
            offsets, resp_start, resp_end, typed_spans, self.dim_to_id)
        num_types = len(self.dim_to_id)
        response_mask = [0] * len(offsets)
        for i, (s, e) in enumerate(offsets):
            if e > s and s >= resp_start and e <= resp_end:
                response_mask[i] = 1

        # LM labels: supervise ONLY the target (rationale + overall) tokens
        lm_labels = [PAD_LABEL] * len(offsets)
        for i, (s, e) in enumerate(offsets):
            if e > s and s >= prompt_len_chars:
                lm_labels[i] = input_ids[i]

        # pad / truncate to max_length
        L = self.max_length
        pad_id = self.tokenizer.pad_token_id
        if len(input_ids) < L:
            pad = L - len(input_ids)
            input_ids = input_ids + [pad_id] * pad
            attention_mask = attention_mask + [0] * pad
            response_mask = response_mask + [0] * pad
            span_labels = span_labels + [[0.0] * num_types] * pad
            lm_labels = lm_labels + [PAD_LABEL] * pad
        else:
            input_ids = input_ids[:L]
            attention_mask = attention_mask[:L]
            response_mask = response_mask[:L]
            span_labels = span_labels[:L]
            lm_labels = lm_labels[:L]

        numeric_targets = [
            float(gold_scores[f]) if gold_scores.get(f) is not None else 3.0
            for f in NUMERIC_SCORE_FIELDS
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "response_mask": torch.tensor(response_mask, dtype=torch.long),
            "span_labels": torch.tensor(span_labels, dtype=torch.float),  # [T, K] multi-hot
            "lm_labels": torch.tensor(lm_labels, dtype=torch.long),
            "numeric_targets": torch.tensor(numeric_targets, dtype=torch.float),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = ["input_ids", "attention_mask", "response_mask", "span_labels", "lm_labels", "numeric_targets"]
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_splits(data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for name in ("train", "dev", "test"):
        path = os.path.join(data_dir, f"{name}.jsonl")
        out[name] = load_jsonl(path) if os.path.exists(path) else []
        logger.info("Loaded %d %s examples from %s", len(out[name]), name, path)
    return out


# -----------------------------------------------------------------------------
# Model: shared Qwen2.5 backbone + score head + span head + native LM head
# -----------------------------------------------------------------------------

class QwenCritic(nn.Module):
    def __init__(self, base_model_name: str, lora_rank: int, lora_alpha: int,
                 lora_dropout: float, torch_dtype: torch.dtype,
                 span_classes: Optional[List[str]] = None,
                 device_index: int = 0, grad_checkpoint: bool = False,
                 attn_impl: str = "flash_attention_2",
                 span_pos_weights: Optional[List[float]] = None,
                 adapter_dir: Optional[str] = None):
        super().__init__()
        self.span_classes = span_classes or ["problem"]
        self.num_span_classes = len(self.span_classes)

        # Load the WHOLE model onto a single GPU. Do NOT use device_map="auto":
        # a 7B/8B model fits on one 48GB L40S, and "auto" would otherwise split it
        # across GPUs (naive pipeline over PCIe) or offload to CPU — both very slow.
        load_kwargs = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                           device_map={"": device_index})
        try:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name, attn_implementation=attn_impl, **load_kwargs)
            logger.info("Loaded base with attn_implementation=%s on cuda:%d", attn_impl, device_index)
        except Exception as e:
            logger.warning("attn_implementation=%s unavailable (%s); falling back to sdpa.",
                           attn_impl, e)
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name, attn_implementation="sdpa", **load_kwargs)

        base.config.use_cache = False
        if grad_checkpoint:
            try:
                base.gradient_checkpointing_enable()
                base.enable_input_require_grads()
                logger.info("Gradient checkpointing enabled (slower, saves memory).")
            except Exception as e:  # pragma: no cover
                logger.warning("Could not enable gradient checkpointing: %s", e)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        if adapter_dir is not None:
            # Inference: attach a TRAINED adapter from a checkpoint instead of a
            # fresh random one. lora_config above is unused in this path (the
            # saved adapter_config.json carries r/alpha/dropout/target_modules).
            from peft import PeftModel
            self.backbone = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        else:
            self.backbone = get_peft_model(base, lora_config)
        self.backbone.print_trainable_parameters()

        hidden = self.backbone.config.hidden_size
        device = next(self.backbone.parameters()).device
        self.score_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, len(NUMERIC_SCORE_FIELDS))
        ).to(device=device, dtype=torch.float32)
        # one sigmoid logit per failure type (multi-label; no "none" class)
        self.span_head = nn.Linear(hidden, self.num_span_classes).to(device=device, dtype=torch.float32)
        # per-type BCE pos_weight (None => uniform). Registered as buffer so it
        # follows .to(device) and DDP broadcast.
        if span_pos_weights is not None:
            self.register_buffer("span_pos_weight",
                                 torch.tensor(span_pos_weights, dtype=torch.float32, device=device))
        else:
            self.span_pos_weight = None

    def forward(self, input_ids, attention_mask, response_mask,
                numeric_targets=None, span_labels=None, lm_labels=None,
                w_score=1.0, w_span=1.0, w_rationale=1.0):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[-1].float()         # [B, T, H]

        # --- score head: mean-pool over response tokens ---
        rm = response_mask.unsqueeze(-1).float()
        pooled = (hidden * rm).sum(dim=1) / rm.sum(dim=1).clamp(min=1.0)
        score_pred = self.score_head(pooled)            # [B, 5]

        # --- span head: per-token, multi-label ---
        span_logits = self.span_head(hidden)            # [B, T, K]

        score_loss = input_ids.new_zeros((), dtype=torch.float32)
        span_loss = input_ids.new_zeros((), dtype=torch.float32)
        rationale_loss = input_ids.new_zeros((), dtype=torch.float32)

        if numeric_targets is not None:
            score_loss = F.mse_loss(score_pred, numeric_targets.float())
        if span_labels is not None:
            # BCE per (token, type); only response tokens count (mask), mean per element
            bce = F.binary_cross_entropy_with_logits(
                span_logits, span_labels.float(),
                pos_weight=self.span_pos_weight, reduction="none")   # [B, T, K]
            m = response_mask.unsqueeze(-1).float()
            span_loss = (bce * m).sum() / (m.sum() * self.num_span_classes).clamp(min=1.0)
        if lm_labels is not None:
            logits = out.logits.float()                 # [B, T, V]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = lm_labels[:, 1:].contiguous()
            rationale_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).long(),
                ignore_index=PAD_LABEL,
            )

        total = w_score * score_loss + w_span * span_loss + w_rationale * rationale_loss
        return {
            "loss": total,
            "loss_score": score_loss.detach(),
            "loss_span": span_loss.detach(),
            "loss_rationale": rationale_loss.detach(),
            "score_pred": score_pred.detach(),
            "span_logits": span_logits.detach(),
        }

    @torch.no_grad()
    def predict(self, tokenizer, question: str, response: str, max_length: int,
                max_new_tokens: int, span_threshold: float) -> Dict[str, Any]:
        """Generate a rationale and predict scores + spans for one example."""
        self.eval()
        device = next(self.backbone.parameters()).device
        prompt, resp_start, resp_end = build_prompt(question, response)

        enc = tokenizer(prompt, add_special_tokens=True, truncation=True,
                        max_length=max_length, return_offsets_mapping=True,
                        return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]

        # heads from a forward pass over the prompt
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1].float()
        rmask = torch.zeros_like(input_ids, dtype=torch.float)
        for i, (s, e) in enumerate(offsets):
            if e > s and s >= resp_start and e <= resp_end:
                rmask[0, i] = 1.0
        pooled = (hidden * rmask.unsqueeze(-1)).sum(dim=1) / rmask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        score_pred = self.score_head(pooled)[0]
        # multi-label span head: each type fires independently when its sigmoid
        # probability clears span_threshold; several types may fire on one token.
        span_prob = torch.sigmoid(self.span_head(hidden))[0]          # [T, K]
        token_pred = (span_prob >= span_threshold).float().tolist()   # [T][K] multi-hot
        pred_spans = decode_multihot_spans(offsets, token_pred, response, resp_start, resp_end, self.span_classes)

        # rationale generation (re-enable cache for fast decoding)
        self.backbone.config.use_cache = True
        gen = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        self.backbone.config.use_cache = False
        gen_text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred_rationale, pred_problems, pred_assessment = parse_generated_critique(gen_text)

        score_dict = {f: float(score_pred[i].item()) for i, f in enumerate(NUMERIC_SCORE_FIELDS)}
        return {
            "scores": score_dict,
            "rationale": pred_rationale,
            "rationale_raw": gen_text,
            "problems": pred_problems,
            "assessment": pred_assessment,
            "span_token_multihot": token_pred,
            "span_spans": pred_spans,
        }


# -----------------------------------------------------------------------------
# Checkpoint
# -----------------------------------------------------------------------------

def save_checkpoint(model: QwenCritic, tokenizer, output_dir: str, tag: str) -> str:
    path = os.path.join(output_dir, tag)
    ensure_dir(path)
    model.backbone.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save({"score_head": model.score_head.state_dict(),
                "span_head": model.span_head.state_dict()},
               os.path.join(path, "heads.pt"))
    logger.info("Checkpoint saved to %s", path)
    return path


def load_critic(checkpoint_dir: str, base_model_name: str, *,
                typed: bool = True, torch_dtype: torch.dtype = torch.bfloat16,
                device_index: int = 0, attn_impl: str = "flash_attention_2",
                target_format: str = "structured"):
    """Reconstruct a trained QwenCritic from a ``save_checkpoint()`` directory.

    The directory holds the PEFT adapter (adapter_config.json + adapter weights),
    the tokenizer, and ``heads.pt`` = {score_head, span_head}. Returns
    ``(model, tokenizer)`` ready for ``model.predict(tokenizer, q, r, ...)``.

    Inference-only: eval mode, no grad. Sets the module-level
    ``ACTIVE_TARGET_FORMAT`` so ``build_prompt`` reproduces the exact task line
    used at training time (default "structured"). Span class count is inferred
    from the saved ``span_head`` width, so 3-way multi-label and 1-way binary
    checkpoints both load.
    """
    from transformers import AutoTokenizer
    global ACTIVE_TARGET_FORMAT
    ACTIVE_TARGET_FORMAT = target_format

    heads = torch.load(os.path.join(checkpoint_dir, "heads.pt"), map_location="cpu")
    K = heads["span_head"]["weight"].shape[0]
    span_classes, _ = span_label_scheme(typed=(K != 1))
    if len(span_classes) != K:                      # scheme/width mismatch -> generic names
        span_classes = [f"type{i}" for i in range(K)]

    # lora_* args are placeholders here: adapter_dir makes __init__ load the
    # trained adapter_config instead of building a fresh LoRA.
    model = QwenCritic(base_model_name, lora_rank=32, lora_alpha=64, lora_dropout=0.05,
                       torch_dtype=torch_dtype, span_classes=span_classes,
                       device_index=device_index, grad_checkpoint=False,
                       attn_impl=attn_impl, adapter_dir=checkpoint_dir)
    device = next(model.backbone.parameters()).device
    model.score_head.load_state_dict(heads["score_head"])
    model.span_head.load_state_dict(heads["span_head"])
    model.score_head.to(device=device, dtype=torch.float32)
    model.span_head.to(device=device, dtype=torch.float32)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# -----------------------------------------------------------------------------
# Evaluation (generation + heads) -> per-example records
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(model: QwenCritic, tokenizer, examples: List[Dict[str, Any]],
                   split_name: str, args, dim_to_id: Dict[str, int],
                   typed: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if args.max_eval_samples > 0:
        examples = examples[: args.max_eval_samples]

    records: List[Dict[str, Any]] = []
    gold_num = {f: [] for f in NUMERIC_SCORE_FIELDS}
    pred_num = {f: [] for f in NUMERIC_SCORE_FIELDS}
    rat_sims: List[Dict[str, float]] = []
    span_classes = sorted(dim_to_id, key=dim_to_id.get)
    span_counts = new_span_counts(len(span_classes))

    for idx, ex in enumerate(examples):
        question = as_str(ex.get("question", ""))
        response = as_str(ex.get("response", ""))
        gold_scores = extract_gold_scores(ex)
        gold_rationale = extract_gold_rationale(ex)

        pred = model.predict(
            tokenizer, question, response,
            max_length=args.max_length, max_new_tokens=args.max_new_tokens,
            span_threshold=args.span_threshold,
        )

        # gold spans -> multi-hot on the same tokenization; pool micro counts
        prompt, rs, re_ = build_prompt(question, response)
        enc = tokenizer(prompt, add_special_tokens=True, truncation=True,
                        max_length=args.max_length, return_offsets_mapping=True)
        offsets = [tuple(x) for x in enc["offset_mapping"]]
        typed_spans = resolve_typed_response_spans(ex, response, gold_rationale)
        gold_mh = typed_char_spans_to_multihot(offsets, rs, re_, typed_spans, dim_to_id)
        valid = [e > s and s >= rs and e <= re_ for (s, e) in offsets]
        K = len(span_classes)
        pred_mh = pred["span_token_multihot"][: len(gold_mh)]
        pred_mh = pred_mh + [[0.0] * K] * (len(gold_mh) - len(pred_mh))
        accumulate_span_counts(span_counts, gold_mh, pred_mh, valid)

        # Structured mode generates problems + assessment separately, so the
        # lexical similarity is computed assessment-vs-overall-gold (comparing
        # against the full multi-dim concat would punish the format change).
        if ACTIVE_TARGET_FORMAT == "structured":
            gold_ref = extract_overall_rationale(ex) or NULL_RATIONALE
        else:
            gold_ref = gold_rationale or NULL_RATIONALE
        sim = rationale_similarity(pred["rationale"], gold_ref)
        rat_sims.append(sim)

        score_err = {}
        for i, f in enumerate(NUMERIC_SCORE_FIELDS):
            g = gold_scores.get(f)
            p = pred["scores"].get(f)
            if g is not None and p is not None:
                gold_num[f].append(float(g))
                pred_num[f].append(float(p))
                score_err[f] = abs(float(g) - float(p))

        records.append({
            "idx": idx,
            "question_id": extract_question_id(ex),
            "topic": extract_topic(ex),
            "question": question,
            "response": response,
            "gold_scores": {f: gold_scores.get(f) for f in NUMERIC_SCORE_FIELDS},
            "pred_scores": {f: round(pred["scores"][f], 4) for f in NUMERIC_SCORE_FIELDS},
            "score_abs_err": {k: round(v, 4) for k, v in score_err.items()},
            "gold_rationale": gold_rationale,
            "pred_rationale": pred["rationale"],
            "pred_rationale_raw": pred["rationale_raw"],
            "gold_problems": extract_problem_items(ex),
            "pred_problems": pred["problems"],
            "rationale_token_f1": round(sim["token_f1"], 4),
            "rationale_rougeL_f1": round(sim["rougeL_f1"], 4),
            "pred_spans": pred["span_spans"],
        })

        if (idx + 1) % args.log_every == 0 or idx == 0:
            logger.info("[%s %d/%d] gold_overall=%s pred_overall=%.2f | rat_tokF1=%.3f",
                        split_name, idx + 1, len(examples),
                        gold_scores.get("overall"), pred["scores"]["overall"], sim["token_f1"])

    summary = {"split": split_name, "n": len(records), "scores": {}, "rationale": {}, "span": {}}
    for f in NUMERIC_SCORE_FIELDS:
        summary["scores"][f] = compute_numeric_metrics(gold_num[f], pred_num[f])
    summary["rationale"] = {
        "token_f1": average_key(rat_sims, "token_f1"),
        "rougeL_f1": average_key(rat_sims, "rougeL_f1"),
    }
    summary["span"] = span_summary_from_counts(span_counts, span_classes)

    def _fmt(x: float) -> str:
        return f"{x:.3f}" if x == x else "nan"

    logger.info("=== %s summary ===", split_name.upper())
    for f in NUMERIC_SCORE_FIELDS:
        m = summary["scores"][f]
        logger.info("  score/%-12s MAE=%.3f spearman=%s", f, m["mae"],
                    f"{m['spearman']:.3f}" if m["spearman"] == m["spearman"] else "nan")
    logger.info("  rationale token_f1=%.3f rougeL=%.3f",
                summary["rationale"]["token_f1"], summary["rationale"]["rougeL_f1"])
    sp = summary["span"]
    logger.info("  span detect (micro) P=%s R=%s F1=%s",
                _fmt(sp["detect_precision"]), _fmt(sp["detect_recall"]), _fmt(sp["detect_f1"]))
    logger.info("  span typed  (micro) P=%s R=%s F1=%s",
                _fmt(sp["typed_precision"]), _fmt(sp["typed_recall"]), _fmt(sp["typed_f1"]))
    for name, m in sp["per_type"].items():
        logger.info("    type/%-9s P=%s R=%s F1=%s (support=%d tokens)",
                    name, _fmt(m["precision"]), _fmt(m["recall"]), _fmt(m["f1"]),
                    m["support_tokens"])
    return summary, records


@torch.no_grad()
def evaluate_dev_loss(model: QwenCritic, loader: DataLoader, device: str, args) -> Dict[str, float]:
    model.eval()
    tot = {"loss": 0.0, "loss_score": 0.0, "loss_span": 0.0, "loss_rationale": 0.0}
    n = 0
    for batch in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**b, w_score=args.score_loss_weight, w_span=args.span_loss_weight,
                    w_rationale=args.rationale_loss_weight)
        tot["loss"] += float(out["loss"].item())
        tot["loss_score"] += float(out["loss_score"].item())
        tot["loss_span"] += float(out["loss_span"].item())
        tot["loss_rationale"] += float(out["loss_rationale"].item())
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in tot.items()}


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(args) -> None:
    global ACTIVE_TARGET_FORMAT
    ACTIVE_TARGET_FORMAT = args.target_format
    # ----- distributed setup (no-op for single-GPU) -----
    distributed, rank, world_size, local_rank = maybe_init_distributed()
    is_main = (rank == 0)
    device_index = local_rank if distributed else args.gpu
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
    # Only rank 0 writes logs / files to avoid concurrent-write corruption.
    if not is_main:
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
        logger.setLevel(logging.WARNING)

    if is_main:
        ensure_dir(args.output_dir)
    set_seed(args.seed)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    if distributed:
        logger.info("DDP enabled: world_size=%d, this rank=%d on cuda:%d", world_size, rank, local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    typed = not args.span_binary
    span_classes, dim_to_id = span_label_scheme(typed)
    logger.info("Span head: %s mode, classes=%s", "typed" if typed else "binary", span_classes)

    splits = load_splits(args.data_dir)

    span_pos_weights = None
    if args.span_weight_cap > 1.0:
        span_pos_weights = compute_span_pos_weights(
            splits["train"], dim_to_id, cap=args.span_weight_cap)
        logger.info("Span BCE pos_weights (cap=%.1f): %s",
                    args.span_weight_cap,
                    {c: round(w, 3) for c, w in zip(span_classes, span_pos_weights)})
    train_ds = CriticDataset(splits["train"], tokenizer, args.max_length, dim_to_id, typed)
    dev_ds = CriticDataset(splits["dev"], tokenizer, args.max_length, dim_to_id, typed)
    train_sampler = (DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
                     if distributed else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, collate_fn=collate_batch,
                              num_workers=args.num_workers, pin_memory=True)
    # dev loss is computed on rank 0 only over the full dev set (no dist sampler).
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_batch, num_workers=args.num_workers, pin_memory=True)

    core = QwenCritic(args.base_model, args.lora_rank, args.lora_alpha,
                      args.lora_dropout, torch_dtype=dtype, span_classes=span_classes,
                      device_index=device_index, grad_checkpoint=args.grad_checkpoint,
                      attn_impl=args.attn_impl, span_pos_weights=span_pos_weights)
    # DDP wraps the whole critic; frozen base params (requires_grad=False) are
    # ignored by DDP, only LoRA + heads are synced. `core` keeps .predict/.save.
    model = DDP(core, device_ids=[local_rank], find_unused_parameters=False) if distributed else core

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_loader) * args.num_epochs) // max(1, args.grad_accum_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    best_dev_loss = float("inf")
    best_path = None
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if distributed:
            train_sampler.set_epoch(epoch)
        ep = {"loss": 0.0, "loss_score": 0.0, "loss_span": 0.0, "loss_rationale": 0.0}
        for step, batch in enumerate(train_loader):
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            is_accum = (step + 1) % args.grad_accum_steps != 0
            # Skip gradient all-reduce on accumulation micro-steps (DDP optimization).
            sync_ctx = model.no_sync() if (distributed and is_accum) else _nullcontext()
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device != "cpu")):
                    out = model(**b, w_score=args.score_loss_weight, w_span=args.span_loss_weight,
                                w_rationale=args.rationale_loss_weight)
                    loss = out["loss"] / args.grad_accum_steps
                # Under DDP every rank must call backward the same number of times,
                # or the gradient all-reduce desyncs and hangs. So only single-GPU
                # runs may skip a non-finite batch; DDP relies on grad clipping.
                if not torch.isfinite(loss) and not distributed:
                    logger.warning("Non-finite loss at epoch %d step %d; skipping.", epoch, step)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()

            for k in ep:
                ep[k] += float(out[k].item())

            if not is_accum:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    logger.info(
                        "epoch %d step %d | loss=%.4f | score=%.4f | span=%.4f | rationale=%.4f",
                        epoch, global_step, float(out["loss"].item()),
                        float(out["loss_score"].item()), float(out["loss_span"].item()),
                        float(out["loss_rationale"].item()))

        nb = max(1, len(train_loader))
        logger.info("epoch %d done | train loss=%.4f score=%.4f span=%.4f rationale=%.4f",
                    epoch, ep["loss"] / nb, ep["loss_score"] / nb,
                    ep["loss_span"] / nb, ep["loss_rationale"] / nb)

        # ----- per-epoch dev loss + checkpoint: rank 0 only; others wait -----
        # We intentionally do NOT run generation-based eval every epoch: generating
        # rationales for all dev examples takes tens of minutes, which exceeds the
        # NCCL collective timeout and aborts the waiting rank. Model selection uses
        # the cheap dev loss (forward only); the full generation eval with rationale
        # metrics runs once at the very end.
        if is_main:
            dev_losses = evaluate_dev_loss(core, dev_loader, device, args)
            logger.info("epoch %d dev | loss=%.4f score=%.4f span=%.4f rationale=%.4f",
                        epoch, dev_losses["loss"], dev_losses["loss_score"],
                        dev_losses["loss_span"], dev_losses["loss_rationale"])
            if dev_losses["loss"] < best_dev_loss:
                best_dev_loss = dev_losses["loss"]
                best_path = save_checkpoint(core, tokenizer, args.output_dir, "best")
                logger.info("New best (dev loss=%.4f)", best_dev_loss)
        if distributed:
            dist.barrier()

    if not is_main:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    final_path = save_checkpoint(core, tokenizer, args.output_dir, "final")
    if best_path is None:
        best_path = final_path

    # final evaluation on dev + test, exported to ONE json file
    logger.info("Final evaluation -> exporting %s", args.results_json)
    final_blob: Dict[str, Any] = {
        "config": vars(args),
        "best_dev_loss": best_dev_loss,
        "metrics": {},
        "predictions": {},
    }
    for split_name in ("dev", "test"):
        if not splits[split_name]:
            continue
        summary, records = evaluate_split(core, tokenizer, splits[split_name], split_name, args, dim_to_id, typed)
        final_blob["metrics"][split_name] = summary
        final_blob["predictions"][split_name] = records

    results_path = args.results_json if os.path.isabs(args.results_json) \
        else os.path.join(args.output_dir, args.results_json)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_blob, f, ensure_ascii=False, indent=2)
    logger.info("All results written to %s", results_path)
    logger.info("Training complete. Best dev loss=%.4f", best_dev_loss)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Qwen2.5 score+span+rationale critic.")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="rationale_critic_checkpoint")
    p.add_argument("--results_json", default="final_results.json",
                   help="Single JSON file with metrics + per-example gold/pred (relative to output_dir).")
    p.add_argument("--log_file", default="train.out")

    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=384,
                   help="Generation cap; structured targets (problems + assessment) "
                        "need more room than the old free-form rationale.")
    p.add_argument("--target_format", choices=["structured", "plain"], default="structured",
                   help="structured: '### Problems:' (span-quoted, typed) + "
                        "'### Overall assessment:' + score; plain: legacy concat rationale.")
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--score_loss_weight", type=float, default=1.0)
    p.add_argument("--span_loss_weight", type=float, default=1.0)
    p.add_argument("--rationale_loss_weight", type=float, default=1.0)

    p.add_argument("--span_threshold", type=float, default=0.5,
                   help="Per-type sigmoid probability above which a failure type fires "
                        "on a token (multi-label head). Lower it to trade precision for recall.")
    p.add_argument("--span_weight_cap", type=float, default=30.0,
                   help="Cap on per-type BCE pos_weight (= neg_chars/pos_chars). "
                        "1.0 disables weighting (uniform BCE).")
    p.add_argument("--span_binary", action="store_true", default=False,
                   help="Single type-agnostic sigmoid (problem/none) instead of the default "
                        "per-type heads (medical/factual/toxicity).")
    p.add_argument("--gpu", type=int, default=0,
                   help="Single GPU index to load the whole model onto (no device_map=auto).")
    p.add_argument("--grad_checkpoint", action="store_true", default=False,
                   help="Enable gradient checkpointing (slower; only needed if you OOM).")
    p.add_argument("--attn_impl", default="flash_attention_2",
                   help="Attention impl: flash_attention_2 (fast, needs flash-attn) or sdpa.")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--bf16", action="store_true", default=True,
                   help="Use bfloat16 (recommended for Qwen2.5). --fp16 to override.")
    p.add_argument("--fp16", dest="bf16", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False))
    train(args)


if __name__ == "__main__":
    main()
