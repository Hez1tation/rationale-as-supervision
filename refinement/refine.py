"""Core critic-guided refinement loop."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .prompts import build_initial_prompt, build_revision_prompt


ISSUE_TERMS = (
    "issue",
    "problem",
    "unsafe",
    "inappropriate",
    "unsupported",
    "medical advice",
    "medication",
    "factual",
    "toxicity",
    "toxic",
    "judgmental",
    "assumption",
    "not empathetic",
    "lacks empathy",
)

NO_ISSUE_TERMS = (
    "no significant issues",
    "no issues identified",
    "no problems identified",
    "no issue identified",
)


@dataclass(frozen=True)
class RefinementConfig:
    max_iter: int = 3
    stop_score: int = 4
    require_no_issue_text: bool = False
    enable_quality_gate: bool = False


ARTIFACT_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "multiple_choice_options",
        re.compile(r"(?:^|\n)\s*A[\).].{0,180}(?:\n|\s+)B[\).].{0,180}(?:\n|\s+)C[\).]", re.DOTALL),
        "The response is formatted like a multiple-choice answer instead of directly responding to the user.",
    ),
    (
        "answer_choice",
        re.compile(r"\bAnswer\s*:\s*[A-D][\).]", re.IGNORECASE),
        "The response contains an explicit answer-choice marker such as 'Answer: A)'.",
    ),
    (
        "reasoning_skill_artifact",
        re.compile(r"\bReasoning\s+Skill\b|\bReasoning\s+skill\b"),
        "The response contains benchmark-style 'Reasoning Skill' text.",
    ),
    (
        "correct_incorrect_answer_artifact",
        re.compile(r"\bcorrect answer\b|\bincorrect answers?\b", re.IGNORECASE),
        "The response discusses correct or incorrect answers rather than counseling the user.",
    ),
    (
        "revision_preamble",
        re.compile(
            r"^\s*(?:here(?:'s| is)\s+(?:the\s+)?(?:revised|rewritten)?\s*response|"
            r"the\s+(?:revised|rewritten)\s+response\s+is|"
            r"this\s+revised\s+response\s+is\s+as\s+follows|"
            r"response|answer|final\s+answer)\s*:",
            re.IGNORECASE,
        ),
        "The response begins with a revision or answer preamble instead of speaking directly to the user.",
    ),
    (
        "embedded_answer_label",
        re.compile(r"(?:^|\n)\s*(?:\*\*)?(?:Response|Final Answer|Answer)(?:\*\*)?\s*:", re.IGNORECASE),
        "The response contains answer labels or section headers instead of only the direct answer.",
    ),
    (
        "instruction_leakage",
        re.compile(
            r"(?:^|\n)\s*-\s*(?:If the user's question|Keep the response|Use a clear|"
            r"Provide empathetic language|Preserve the original|Use a conversational tone)",
            re.IGNORECASE,
        ),
        "The response appears to repeat output instructions instead of answering the user.",
    ),
    (
        "meta_explanation",
        re.compile(
            r"\b(?:This response addresses|This answer addresses|The response is written|"
            r"It provides a clear and concise answer|By doing so, it creates|"
            r"This revised response)\b",
            re.IGNORECASE,
        ),
        "The response explains or evaluates itself instead of only answering the user.",
    ),
)


def critique_has_issue_language(rationale: str) -> bool:
    lowered = rationale.lower()
    if any(term in lowered for term in NO_ISSUE_TERMS):
        return False
    return any(term in lowered for term in ISSUE_TERMS)


def detect_response_artifacts(response: str) -> list[dict[str, str]]:
    """Find deterministic formatting artifacts that the critic often misses."""
    artifacts: list[dict[str, str]] = []
    for artifact_type, pattern, message in ARTIFACT_PATTERNS:
        if pattern.search(response):
            artifacts.append({"type": artifact_type, "message": message})
    return artifacts


def apply_quality_gate(critique: dict[str, Any], response: str, enabled: bool = True) -> dict[str, Any]:
    """Cap critic score when deterministic response-quality artifacts are present."""
    if not enabled:
        return critique

    artifacts = detect_response_artifacts(response)
    if not artifacts:
        return critique

    gated = dict(critique)
    original_score = gated.get("score")
    gated["original_score"] = original_score
    gated["score"] = min(int(original_score) if original_score is not None else 2, 2)
    gated["quality_gate"] = {
        "triggered": True,
        "artifacts": artifacts,
        "score_cap": 2,
    }

    gate_rationale = (
        "Deterministic quality gate triggered: the response appears to contain "
        "benchmark or multiple-choice artifacts rather than a direct, supportive, "
        "clinically cautious counseling response. Rewrite it as a normal response "
        "to the user without answer choices, grading language, revision preambles, "
        "instruction leakage, meta-explanations, or reasoning-skill labels."
    )
    existing = (gated.get("rationale") or "").strip()
    gated["rationale"] = f"{existing}\n\n{gate_rationale}".strip()

    raw_output = (gated.get("raw_output") or "").strip()
    gated["raw_output"] = (
        f"{raw_output}\n\n### Quality Gate:\n{gate_rationale}".strip()
    )
    return gated


def validate_final_response(response: str) -> dict[str, Any]:
    """Evaluate final output formatting without changing the response or score."""
    artifacts = detect_response_artifacts(response)
    return {
        "is_clean": not artifacts,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }


def should_stop(critique: dict[str, Any], config: RefinementConfig) -> tuple[bool, str]:
    """Stop when the critic is satisfied: score >= stop_score OR it explicitly
    says no issues (schedule.pdf Stage 4 termination).

    The old `has_issue_language` keyword gate is dropped: our critic's rationale
    enumerates every failure dimension by name (empathy / factual / medication /
    assumption / toxicity ...), so that gate matched on essentially every
    critique and the loop never terminated early -- it always ran to max_iter.
    `require_no_issue_text=True` keeps a stricter AND mode (score high AND the
    critic states no issues) for callers who want it.
    """
    score = critique.get("score")
    rationale = (critique.get("rationale") or "").lower()
    no_issue_text = any(term in rationale for term in NO_ISSUE_TERMS)
    score_ok = score is not None and score >= config.stop_score

    if config.require_no_issue_text:
        if score_ok and no_issue_text:
            return True, "score_and_no_issues"
        return False, ""
    if score_ok:
        return True, "score_threshold"
    if no_issue_text:
        return True, "no_issues"
    return False, ""


def critic_guided_generate(
    question: str,
    generator,
    critic,
    config: RefinementConfig | None = None,
) -> dict[str, Any]:
    """Generate and iteratively refine one response."""
    config = config or RefinementConfig()
    initial_prompt = build_initial_prompt(question)
    response = generator.generate(initial_prompt)
    initial_response = response
    iterations: list[dict[str, Any]] = []
    final_score: float | None = None
    final_critique: dict[str, Any] | None = None
    stop_reason = "max_iter"

    for iteration in range(1, config.max_iter + 1):
        critique = apply_quality_gate(
            critic.critique(question, response),
            response,
            enabled=config.enable_quality_gate,
        )
        stop, reason = should_stop(critique, config)
        if stop:
            # The current response satisfied the critic; we keep it unchanged,
            # so this critique describes the final response. final_score matches.
            stop_reason = reason
            final_score = critique.get("score")
            final_critique = critique
            iterations.append(
                {
                    "iteration": iteration,
                    "response_before": response,
                    "critique": critique,
                    "revision_prompt": None,
                    "response_after": response,
                    "stopped": True,
                }
            )
            break

        revision_prompt = build_revision_prompt(question, response, critique)
        revised_response = generator.generate(revision_prompt)
        iterations.append(
            {
                "iteration": iteration,
                "response_before": response,
                "critique": critique,
                "revision_prompt": revision_prompt,
                "response_after": revised_response,
                "stopped": False,
            }
        )
        response = revised_response
    else:
        # max_iter exhausted without satisfying the critic. The last revision was
        # never scored, so critique the actual final response once more -- otherwise
        # final_score describes the pre-revision draft, not final_response.
        final_critique = apply_quality_gate(
            critic.critique(question, response),
            response,
            enabled=config.enable_quality_gate,
        )
        final_score = final_critique.get("score")

    return {
        "initial_prompt": initial_prompt,
        "initial_response": initial_response,
        "iterations": iterations,
        "final_response": response,
        "final_score": final_score,
        "final_critique": final_critique,
        "final_validation": validate_final_response(response),
        "stop_reason": stop_reason,
    }
