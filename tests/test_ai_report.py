import sys
from types import SimpleNamespace

import pytest

from src.reporting.ai_report import generate_report


def sample_metrics():
    return {
        "selected_model": "XGBoost",
        "poisson_deviance": 0.581796,
        "normalized_gini": 0.331094,
        "pure_premium_ae": 0.993015,
        "worst_segment": "vehicle brand B11",
        "worst_segment_loss_ratio": 3.778551,
    }


def test_deterministic_report_contains_supplied_metrics():
    report = generate_report(
        sample_metrics(),
        use_llm=False,
    )

    assert "XGBoost" in report
    assert "0.5818" in report
    assert "0.3311" in report
    assert "0.993" in report
    assert "vehicle brand B11" in report
    assert "377.9%" in report
    assert "277.9% above break-even" in report


def test_generate_report_rejects_missing_metrics():
    metrics = sample_metrics()
    del metrics["normalized_gini"]

    with pytest.raises(
        ValueError,
        match="Missing report metrics",
    ):
        generate_report(
            metrics,
            use_llm=False,
        )


def test_llm_mode_without_api_key_uses_deterministic_fallback(monkeypatch):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    report = generate_report(
        sample_metrics(),
        use_llm=True,
    )

    assert "XGBoost" in report
    assert "vehicle brand B11" in report
    assert "377.9%" in report


def test_llm_api_error_uses_deterministic_fallback(monkeypatch):
    class FakeOpenAIError(Exception):
        pass

    class FakeResponses:
        def create(self, **kwargs):
            raise FakeOpenAIError("no credits")

    class FakeClient:
        responses = FakeResponses()

    fake_openai = SimpleNamespace(
        OpenAI=lambda api_key: FakeClient(),
        OpenAIError=FakeOpenAIError,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    report = generate_report(sample_metrics(), use_llm=True)

    assert "XGBoost" in report
    assert "vehicle brand B11" in report
