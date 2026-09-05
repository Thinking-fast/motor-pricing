from pathlib import Path

import run_pipeline


def test_ensure_database_reuses_populated_database(monkeypatch):
    monkeypatch.setattr(run_pipeline, "database_is_ready", lambda db_url: True)

    config = {
        "database": {"url": "sqlite:///example.db"},
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
        },
        "_project_root": ".",
    }

    assert run_pipeline.ensure_database(config) is False


def test_ensure_database_builds_missing_database(monkeypatch, tmp_path):
    frequency_path = tmp_path / "frequency.csv"
    severity_path = tmp_path / "severity.csv"
    frequency_path.write_text("IDpol,Exposure\n1,1.0\n", encoding="utf-8")
    severity_path.write_text("IDpol,ClaimAmount\n1,100\n", encoding="utf-8")

    loaded = {}
    monkeypatch.setattr(run_pipeline, "database_is_ready", lambda db_url: False)
    monkeypatch.setattr(
        run_pipeline,
        "download_fremtpl2",
        lambda raw_directory: (frequency_path, severity_path),
    )

    def fake_load_to_db(freq, sev, db_url, schema_sql):
        loaded["frequency_rows"] = len(freq)
        loaded["severity_rows"] = len(sev)
        loaded["db_url"] = db_url
        loaded["schema_sql"] = schema_sql

    monkeypatch.setattr(run_pipeline, "load_to_db", fake_load_to_db)

    config = {
        "database": {"url": "sqlite:///example.db"},
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
        },
        "_project_root": str(tmp_path),
    }

    assert run_pipeline.ensure_database(config) is True
    assert loaded["frequency_rows"] == 1
    assert loaded["severity_rows"] == 1
    assert loaded["db_url"] == "sqlite:///example.db"
    assert loaded["schema_sql"] == Path(tmp_path) / "sql" / "schema.sql"
    assert (tmp_path / "data" / "processed").is_dir()


def test_run_pipeline_calls_stages_in_order(monkeypatch):
    calls = []
    config = {"database": {"url": "sqlite:///example.db"}}

    monkeypatch.setattr(run_pipeline, "load_config", lambda: config)
    monkeypatch.setattr(
        run_pipeline,
        "ensure_database",
        lambda config, force_reload: calls.append("database") or False,
    )
    monkeypatch.setattr(
        run_pipeline,
        "run_experience_studies",
        lambda: calls.append("experience") or {},
    )
    monkeypatch.setattr(
        run_pipeline,
        "run_technical_premium",
        lambda: calls.append("premium") or {},
    )
    monkeypatch.setattr(
        run_pipeline,
        "run_frequency_models",
        lambda: calls.append("models") or {},
    )
    monkeypatch.setattr(
        run_pipeline,
        "run_report",
        lambda: calls.append("report") or "summary",
    )

    result = run_pipeline.run_pipeline()

    assert calls == ["database", "experience", "premium", "models", "report"]
    assert result["database_rebuilt"] is False
    assert result["report"] == "summary"
