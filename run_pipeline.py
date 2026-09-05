"""Run the complete motor-pricing platform with one command."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Callable, TypeVar

import pandas as pd
from sqlalchemy import create_engine, inspect

from src.config import load_config
from src.data.download import download_fremtpl2
from src.data.load_to_db import load_to_db
from src.pipelines.run_experience_study import run_experience_studies
from src.pipelines.run_models import run_frequency_models
from src.pipelines.run_report import run_report
from src.pipelines.run_technical_premium import run_technical_premium

logger = logging.getLogger(__name__)

RESULT = TypeVar("RESULT")
REQUIRED_TABLES = {"policies", "claims", "regions"}


def database_is_ready(db_url: str) -> bool:
    """Return whether the configured database contains populated core tables."""
    try:
        engine = create_engine(db_url)
        table_names = set(inspect(engine).get_table_names())
        if not REQUIRED_TABLES.issubset(table_names):
            return False

        with engine.connect() as connection:
            policy_count = connection.exec_driver_sql(
                "SELECT COUNT(*) FROM policies"
            ).scalar_one()
            claim_count = connection.exec_driver_sql(
                "SELECT COUNT(*) FROM claims"
            ).scalar_one()
        return policy_count > 0 and claim_count > 0
    except Exception as error:
        logger.warning("Database readiness check failed: %s", error)
        return False


def ensure_database(config: dict, force_reload: bool = False) -> bool:
    """Build the SQL database when missing, invalid, or explicitly requested."""
    db_url = config["database"]["url"]
    if not force_reload and database_is_ready(db_url):
        logger.info("Database is populated; reusing %s", db_url)
        return False

    project_root = Path(config["_project_root"])
    raw_directory = project_root / config["paths"]["raw_data"]
    processed_directory = project_root / config["paths"]["processed_data"]
    schema_path = project_root / "sql" / "schema.sql"

    # SQLite cannot create a database when its parent directory is absent.
    processed_directory.mkdir(parents=True, exist_ok=True)

    frequency_path, severity_path = download_fremtpl2(raw_directory)
    logger.info("Reading cached source CSV files")
    frequency = pd.read_csv(frequency_path)
    severity = pd.read_csv(severity_path)

    load_to_db(
        freq=frequency,
        sev=severity,
        db_url=db_url,
        schema_sql=schema_path,
    )
    return True


def run_stage(name: str, function: Callable[[], RESULT]) -> RESULT:
    """Run one pipeline stage with consistent timing and logging."""
    logger.info("Starting stage: %s", name)
    start = perf_counter()
    result = function()
    logger.info("Completed stage: %s in %.1f seconds", name, perf_counter() - start)
    return result


def run_pipeline(force_reload: bool = False) -> dict:
    """Run database, analysis, pricing, modelling and reporting stages."""
    config = load_config()
    started = perf_counter()
    logger.info("Starting motor-pricing pipeline")

    database_rebuilt = run_stage(
        "source data and SQL database",
        lambda: ensure_database(config, force_reload=force_reload),
    )
    experience = run_stage("experience studies", run_experience_studies)
    technical_premium = run_stage("technical premium", run_technical_premium)
    models = run_stage("pricing models", run_frequency_models)
    report = run_stage("management report", run_report)

    elapsed = perf_counter() - started
    logger.info("Pipeline completed successfully in %.1f seconds", elapsed)
    logger.info(
        "Launch the dashboard with: python -m streamlit run app/streamlit_app.py"
    )

    return {
        "database_rebuilt": database_rebuilt,
        "experience": experience,
        "technical_premium": technical_premium,
        "models": models,
        "report": report,
        "elapsed_seconds": elapsed,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="rebuild the SQL database even when it is already populated",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    arguments = parse_args()
    run_pipeline(force_reload=arguments.force_reload)
