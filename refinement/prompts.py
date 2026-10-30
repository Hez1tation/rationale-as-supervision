"""Prompt templates for Stage 4 refinement."""

from __future__ import annotations

from typing import Any


FAILURE_MODE_DEFINITIONS = (
    "- empathy: validate the user's feelings and avoid apathetic or dismissive language.\n"
    "- specificity: tailor the response to the user's stated situation; avoid generic advice "
    "or unsupported assumptions.\n"
    "- medical_advice: identify therapeutic or diagnostic guidance that should only be "
    "provided by licensed professionals.\n"
    "- factual_consistency: avoid unsupported clinical claims and keep statements aligned with "
    "general mental-health knowledge.\n"
    "- toxicity: avoid stigmatizing, judgmental, shaming, coercive, or otherwise harmful language."
)


OUTPUT_FORMAT_RULES = (
    "Output format (important):\n"
    "- Write ONLY the response itself, addressed directly to the user.\n"
    "- Do NOT add any preamble, heading, or label such as 'Response:', "
    "'Here is the response:', 'Answer:', or 'Revised response:'.\n"
    "- Do NOT wrap the response in quotation marks.\n"
    "- Do NOT add any commentary, evaluation, or explanation about the response "
    "(no 'This response aims to...').\n"
    "- Do NOT use placeholders like [User], [Name], or [Your name]; speak to the "
    "person directly."
)


def build_initial_prompt(question: str) -> str:
    """First-pass prompt: identity + the user's message only.

    Deliberately does NOT hand the generator the critic's quality dimensions.
    Listing the exact rubric the response will be scored on is "teaching to the
    test": the model then writes to cover empathy/specificity/safety on cue, so
    every response saturates the critic's high band (v2: 120/120 scored 4-5,
    std 0.22) and the refinement loop has no gradient to climb. A real counselor
    answers the person, not a checklist -- this keeps the initial generation
    naturalistic so the critic has room to discriminate. OUTPUT_FORMAT_RULES stay
    because they only govern formatting (no preamble/labels), which is orthogonal
    to content quality; dropping them brings back the 53% artifact rate.
    """
    return (
        "You are a mental-health counselor. Someone has reached out to you for support.\n\n"
        "Their message:\n"
        f"{question.strip()}\n\n"
        "Write your response to them.\n\n"
        f"{OUTPUT_FORMAT_RULES}"
    )


def _span_label(span: dict[str, Any]) -> str:
    return (
        span.get("failure_type")
        or span.get("label")
        or span.get("type")
        or span.get("category")
        or "issue"
    )


def _span_text(span: dict[str, Any]) -> str:
    text = span.get("text") or span.get("sentence") or span.get("copy")
    if text:
        return str(text).strip()
    start = span.get("start_char")
    end = span.get("end_char")
    if start is not None and end is not None:
        return f"char {start}-{end}"
    return "unspecified location"


def _flatten_span_labels(span_labels: dict[str, Any]) -> list[dict[str, str]]:
    spans: list[dict[str, str]] = []
    for label, items in span_labels.items():
        if label == "no_failure" or not items:
            continue
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if isinstance(item, dict):
                text = _span_text(item)
            else:
                text = str(item).strip()
            spans.append({"label": str(label), "text": text})
    return spans


def _format_rationale_by_type(rationale: Any, active_labels: set[str]) -> str:
    if isinstance(rationale, dict):
        parts = []
        for label in ("empathy", "specificity", "medical_advice", "factual_consistency", "toxicity"):
            value = rationale.get(label)
            if not value:
                continue
            if active_labels and label not in active_labels:
                continue
            parts.append(f"[{label}]\n{str(value).strip()}")
        overall = rationale.get("overall")
        if overall:
            parts.append("[overall]\n" + str(overall).strip())
        return "\n\n".join(parts).strip()
    return str(rationale or "").strip()


def format_critique_for_revision(critique: dict[str, Any]) -> str:
    """Render structured critique fields into revision prompt text."""
    pieces: list[str] = []
    score = critique.get("score")
    if score is not None:
        pieces.append(f"Overall critic score: {score}/5")

    spans = critique.get("spans") or []
    if not spans and isinstance(critique.get("span_labels"), dict):
        spans = _flatten_span_labels(critique["span_labels"])

    active_labels: set[str] = set()
    if spans:
        span_lines = []
        for span in spans:
            if isinstance(span, dict):
                label = _span_label(span)
                text = _span_text(span)
            else:
                label = "issue"
                text = str(span).strip()
            active_labels.add(str(label))
            span_lines.append(f"- {label}: \"{text}\"")
        pieces.append("Flagged response parts:\n" + "\n".join(span_lines))
    else:
        pieces.append("Flagged response parts:\n(none provided by the critic)")

    rationale = _format_rationale_by_type(critique.get("rationale"), active_labels)
    if rationale:
        pieces.append("Why these are problems:\n" + rationale)

    raw_output = (critique.get("raw_output") or "").strip()
    if not pieces and raw_output:
        pieces.append("Raw critic output:\n" + raw_output)

    return "\n\n".join(pieces).strip() or "The critic found issues, but did not provide details."


def build_revision_prompt(question: str, response: str, critique: dict[str, Any]) -> str:
    """Build a critique-conditioned revision prompt for the generator."""
    rendered_critique = format_critique_for_revision(critique)
    return (
        "You are a mental-health support response editor. Your task is to fix the "
        "specific problems identified by a clinical-safety critic while preserving "
        "the useful parts of the draft.\n\n"
        "Failure mode definitions:\n"
        f"{FAILURE_MODE_DEFINITIONS}\n\n"
        "Original question:\n"
        f"{question.strip()}\n\n"
        "Draft response to revise:\n"
        f"{response.strip()}\n\n"
        "Critic feedback:\n"
        f"{rendered_critique}\n\n"
        "Revision instructions:\n"
        "1. Rewrite the flagged or criticized parts so the identified failure modes are "
        "resolved.\n"
        "2. Preserve uncriticized content as much as possible; adjust only what is needed "
        "for coherence with the revised text.\n"
        "3. Use the failure mode definitions and critic rationale above as the basis for "
        "the revision.\n\n"
        f"{OUTPUT_FORMAT_RULES}"
    )
