"""Critic wrappers for Stage 4 refinement."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Protocol


DEFAULT_STAGE3_CODE_DIR = Path(
    "/project2/ruishanl_1185/rationale_as_supervision/Pipeline/scripts"
)
DEFAULT_STAGE3_CHECKPOINT = Path(
    "/project2/ruishanl_1185/rationale_as_supervision/Pipeline/checkpoints/rationale_critic"
)


class Critic(Protocol):
    def critique(self, question: str, response: str) -> dict[str, Any]:
        ...


def normalize_critique(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize Stage 3 critic outputs into the Stage 4 schema."""
    score = raw.get("score", raw.get("overall_score"))
    if score is None:
        score = parse_score(raw.get("raw_output", ""))
    if score is not None:
        score = int(round(float(score)))
        score = max(1, min(5, score))

    return {
        "spans": raw.get("spans") or raw.get("span_labels") or [],
        "rationale": (raw.get("rationale") or "").strip(),
        "score": score,
        "raw_output": raw.get("raw_output", ""),
    }


def parse_score(text: str) -> int | None:
    match = re.search(r"###\s*Overall:\s*([1-5])\s*(?:/\s*5)?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\bOverall\s*:\s*([1-5])\s*(?:/\s*5)?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


class PipelineStage3Critic:
    """Adapter around Pipeline/scripts/stage3_generate.py CriticInference."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_STAGE3_CHECKPOINT,
        code_dir: str | Path = DEFAULT_STAGE3_CODE_DIR,
        device: str | None = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 512,
    ) -> None:
        code_path = Path(code_dir)
        if not code_path.exists():
            raise FileNotFoundError(f"Missing Stage 3 code directory: {code_path}")
        if str(code_path) not in sys.path:
            sys.path.insert(0, str(code_path))

        from stage3_generate import CriticInference

        self._critic = CriticInference(
            checkpoint=Path(checkpoint),
            device=device,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
        )

    def critique(self, question: str, response: str) -> dict[str, Any]:
        return normalize_critique(self._critic.critique(question, response))


class LocalRationaleCritic:
    """Adapter around this repo's train_critic.QwenCritic (Stage 3 critic).

    Loads a save_checkpoint() directory via train_critic.load_critic and maps
    QwenCritic.predict() output into the Stage 4 critique schema. Differences
    from PipelineStage3Critic (which text-parses a separate codebase):
      * score is the regression head's float `overall` (NOT regex'd from text,
        NOT rounded to int) -- the writeup deems the head more reliable;
      * spans come from the multi-label span head (`span_spans`), already
        carrying `failure_type` + `text`, which prompts.py renders directly.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        base_model: str,
        code_dir: str | Path | None = None,
        device_index: int = 0,
        max_length: int = 1024,
        max_new_tokens: int = 384,
        span_threshold: float = 0.5,
        target_format: str = "structured",
    ) -> None:
        # train_critic.py lives at the project root; add it to sys.path when the
        # refinement package is imported from elsewhere on the server.
        if code_dir is not None and str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))

        import train_critic as tc

        self._tc = tc
        self.model, self.tokenizer = tc.load_critic(
            str(checkpoint),
            base_model,
            device_index=device_index,
            target_format=target_format,
        )
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.span_threshold = span_threshold

    def critique(self, question: str, response: str) -> dict[str, Any]:
        out = self.model.predict(
            self.tokenizer,
            question,
            response,
            max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            span_threshold=self.span_threshold,
        )
        scores = out.get("scores") or {}
        return {
            "score": scores.get("overall"),          # float, regression head
            "scores": scores,                          # all 5 dims, for analysis
            "rationale": (out.get("rationale") or "").strip(),
            "spans": out.get("span_spans") or [],      # dicts: failure_type + text
            "raw_output": out.get("rationale_raw") or "",
        }


class MockCritic:
    """Small deterministic critic for dry-run testing without model loading."""

    def critique(self, question: str, response: str) -> dict[str, Any]:
        lowered = response.lower()
        risky_terms = ["diagnose", "medication", "prescribe", "you should take"]
        if any(term in lowered for term in risky_terms):
            score = 2
            rationale = (
                "The response may provide overly specific clinical or medication advice. "
                "It should be reframed as general support and encourage professional care."
            )
        elif len(response.split()) < 35:
            score = 3
            rationale = (
                "The response is safe but too brief. It should add empathy, concrete coping "
                "steps, and appropriate support-seeking guidance."
            )
        else:
            score = 4
            rationale = "No significant issues identified."
        return {
            "spans": [],
            "rationale": rationale,
            "score": score,
            "raw_output": f"### Rationale:\n{rationale}\n\n### Overall: {score}/5",
        }


def build_critic(
    mode: str,
    checkpoint: str | Path | None = None,
    code_dir: str | Path | None = None,
    device: str | None = None,
    load_in_4bit: bool = False,
    max_new_tokens: int = 512,
    base_model: str | None = None,
    device_index: int = 0,
) -> Critic:
    if mode == "mock":
        return MockCritic()
    if mode == "pipeline":
        return PipelineStage3Critic(
            checkpoint=checkpoint or DEFAULT_STAGE3_CHECKPOINT,
            code_dir=code_dir or DEFAULT_STAGE3_CODE_DIR,
            device=device,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
        )
    if mode in ("ours", "local"):
        if not checkpoint or not base_model:
            raise ValueError(
                "mode='ours' needs checkpoint=<save_checkpoint dir, e.g. "
                "rationale_critic_llama_checkpoint/final> and base_model=<HF id "
                "used for training, e.g. NousResearch/Meta-Llama-3.1-8B-Instruct>"
            )
        return LocalRationaleCritic(
            checkpoint=checkpoint,
            base_model=base_model,
            code_dir=code_dir,
            device_index=device_index,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unsupported critic mode: {mode}")
