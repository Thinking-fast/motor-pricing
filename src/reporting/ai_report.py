"""Generate management narratives from validated model metrics."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


REQUIRED_METRICS = {
    "selected_model",
    "poisson_deviance",
    "normalized_gini",
    "pure_premium_ae",
    "worst_segment",
    "worst_segment_loss_ratio",
}


def validate_metrics(metrics: dict) -> None:
    """Check that all required report metrics are present."""

    missing = REQUIRED_METRICS - set(metrics)

    if missing:
        raise ValueError(f"Missing report metrics: {sorted(missing)}")


def deterministic_report(metrics: dict) -> str:
    """Generate a management summary without using an LLM."""

    validate_metrics(metrics)

    loss_ratio = metrics["worst_segment_loss_ratio"]
    above_break_even = (loss_ratio - 1) * 100

    return (
        f"The selected frequency model is "
        f"{metrics['selected_model']}, with test Poisson deviance "
        f"{metrics['poisson_deviance']:.4f} and normalized Gini "
        f"{metrics['normalized_gini']:.4f}. "
        f"The combined pure-premium model has an actual-to-expected "
        f"ratio of {metrics['pure_premium_ae']:.3f}. "
        f"The least profitable credible cohort is "
        f"{metrics['worst_segment']}, with a loss ratio of "
        f"{loss_ratio:.1%}, which is {above_break_even:.1f}% "
        f"above break-even. Consider reviewing pricing and experience "
        f"for this cohort."
    )


def llm_report(
    metrics: dict,
    model: str,
) -> str:
    """Ask OpenAI to turn trusted metrics into management prose."""

    validate_metrics(metrics)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for LLM reporting")

    from openai import OpenAI, OpenAIError

    client = OpenAI(api_key=api_key)

    instructions = (
        "Write one concise management paragraph about a motor-insurance "
        "pricing analysis. Use only the supplied metrics. Do not invent, "
        "estimate, derive, or alter any figures. Clearly distinguish a "
        "loss ratio from the percentage above break-even. State that "
        "premium is constructed because the source data has no observed "
        "premium."
    )

    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(metrics, indent=2),
        )
    except OpenAIError as exc:
        logger.warning(
            "OpenAI request failed (%s); using deterministic report",
            type(exc).__name__,
        )
        return deterministic_report(metrics)

    return response.output_text.strip()


def generate_report(
    metrics: dict,
    use_llm: bool = False,
    model: str = "gpt-4o-mini",
) -> str:
    """Generate a deterministic or LLM-written report."""

    if use_llm:
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY is not set; using deterministic report")
            return deterministic_report(metrics)

        logger.info(
            "Generating management narrative using %s",
            model,
        )

        return llm_report(
            metrics,
            model=model,
        )

    logger.info("Generating deterministic management narrative")

    return deterministic_report(metrics)
