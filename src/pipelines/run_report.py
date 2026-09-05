"""Generate the management pricing report."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.reporting.ai_report import generate_report

logger = logging.getLogger(__name__)


def load_report_metrics(
    processed_directory: Path,
) -> dict:
    """Load validated modelling outputs for reporting."""

    frequency = pd.read_csv(processed_directory / "frequency_model_metrics.csv")

    pure_premium = pd.read_csv(processed_directory / "pure_premium_model_metrics.csv")

    profitability = pd.read_csv(
        processed_directory / "modelled_profitability_veh_brand.csv"
    )

    xgb_frequency = frequency.loc[frequency["model"] == "xgboost"].iloc[0]

    xgb_pure_premium = pure_premium.loc[pure_premium["model"] == "xgboost"].iloc[0]

    credible = profitability.loc[profitability["credible"].astype(bool)]

    worst = credible.sort_values(
        "loss_ratio",
        ascending=False,
    ).iloc[0]

    return {
        "selected_model": "XGBoost",
        "poisson_deviance": float(xgb_frequency["poisson_deviance"]),
        "normalized_gini": float(xgb_frequency["normalized_gini"]),
        "pure_premium_ae": float(xgb_pure_premium["actual_to_expected"]),
        "worst_segment": (f"vehicle brand {worst['veh_brand']}"),
        "worst_segment_loss_ratio": float(worst["loss_ratio"]),
    }


def run_report() -> str:
    """Generate and save the management summary."""

    config = load_config()

    processed_directory = Path(config["paths"]["processed_data"])

    report_directory = Path(config["paths"]["reports"])

    metrics = load_report_metrics(processed_directory)

    report = generate_report(
        metrics,
        use_llm=config["reporting"]["use_llm"],
        model=config["reporting"]["llm_model"],
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = report_directory / "management_summary.md"

    output_path.write_text(
        report + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Saved management report to %s",
        output_path,
    )

    print(report)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_report()
