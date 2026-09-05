"""Interactive dashboard for the motor-pricing analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import load_config

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "reports" / "management_summary.md"

RISK_FACTORS = {
    "Driver age": "age_band",
    "Region": "region",
    "Vehicle brand": "veh_brand",
    "Fuel type": "veh_gas",
    "Area": "area",
}

MODEL_LABELS = {
    "constant_baseline": "Constant baseline",
    "poisson_glm": "Poisson GLM",
    "xgboost": "XGBoost",
}


st.set_page_config(
    page_title="Motor Pricing Platform",
    page_icon="🚗",
    layout="wide",
)


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    """Load one generated dashboard dataset."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dashboard data: {path}. Run the modelling pipelines first."
        )
    return pd.read_csv(path)


@st.cache_data
def load_management_summary() -> str | None:
    """Load the generated narrative when it is available."""
    if not REPORT_PATH.exists():
        return None
    return REPORT_PATH.read_text(encoding="utf-8").strip()


def model_names(series: pd.Series) -> pd.Series:
    """Convert internal model identifiers into presentation labels."""
    return series.map(MODEL_LABELS).fillna(series)


def render_executive_view(config: dict) -> None:
    """Render decision-focused portfolio results."""
    st.header("Executive summary")
    st.write(
        "The selected model differentiates claim frequency and converts expected "
        "cost into a constructed technical premium."
    )

    predictions = load_csv("pure_premium_xgboost_predictions.csv")
    brand_profitability = load_csv("modelled_profitability_veh_brand.csv")

    total_claims = predictions["total_claim_amount"].sum()
    predicted_claim_cost = predictions["predicted_claim_cost"].sum()
    loading_factor = (
        1 + config["pricing"]["expense_loading"] + config["pricing"]["profit_loading"]
    )
    total_technical_premium = predicted_claim_cost * loading_factor
    overall_loss_ratio = total_claims / total_technical_premium
    overall_ae = total_claims / predicted_claim_cost

    credible_brands = brand_profitability.loc[brand_profitability["credible"]]
    worst_brand = credible_brands.sort_values("loss_ratio", ascending=False).iloc[0]

    column1, column2, column3, column4 = st.columns(4)
    column1.metric("Constructed premium", f"€{total_technical_premium:,.0f}")
    column2.metric("Actual claims", f"€{total_claims:,.0f}")
    column3.metric("Overall loss ratio", f"{overall_loss_ratio:.1%}")
    column4.metric("Pure-premium A/E", f"{overall_ae:.3f}")

    st.info(
        "Premium is constructed from predicted claim cost plus expense and profit "
        "loadings. It is not premium observed in the source dataset."
    )

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Management commentary")
        narrative = load_management_summary()
        if narrative:
            st.write(narrative)
        else:
            st.warning(
                "Management commentary is unavailable. Run "
                "`python -m src.pipelines.run_report`."
            )

    with right:
        st.subheader("Priority cohort")
        st.metric(
            f"Vehicle brand {worst_brand['veh_brand']}",
            f"{worst_brand['loss_ratio']:.1%}",
            delta=f"{worst_brand['loss_ratio'] - 1:.1%} vs break-even",
            delta_color="inverse",
        )
        st.caption(
            "Highest loss ratio among credible vehicle-brand cohorts in the "
            "held-out test sample."
        )


def render_risk_view() -> None:
    """Render one-way actuarial risk and profitability analysis."""
    st.header("Risk analysis")
    st.write(
        "Select one rating factor to see how profitability, frequency and severity "
        "vary across portfolio segments."
    )

    control1, control2 = st.columns(2)
    factor_label = control1.selectbox("Rating factor", options=list(RISK_FACTORS))
    loss_basis = control2.radio(
        "Experience basis",
        options=["Capped", "Gross"],
        horizontal=True,
        help="Capped experience limits each policy's aggregate claim amount to €100,000.",
    )
    factor = RISK_FACTORS[factor_label]

    profitability = load_csv(f"modelled_profitability_{factor}.csv")
    experience_suffix = "_capped" if loss_basis == "Capped" else ""
    experience = load_csv(f"experience_{factor}{experience_suffix}.csv")
    category_order = experience[factor].astype(str).tolist()

    loss_ratio_chart = px.bar(
        profitability,
        x=factor,
        y="loss_ratio",
        color="credible",
        category_orders={factor: category_order},
        title=f"Constructed loss ratio by {factor_label.lower()}",
        labels={
            factor: factor_label,
            "loss_ratio": "Loss ratio",
            "credible": "Credible exposure",
        },
    )
    loss_ratio_chart.add_hline(
        y=1,
        line_dash="dash",
        line_color="red",
        annotation_text="Break-even",
    )
    loss_ratio_chart.update_yaxes(tickformat=".0%")
    st.plotly_chart(loss_ratio_chart, width="stretch")

    chart1, chart2 = st.columns(2)
    frequency_chart = px.bar(
        experience,
        x=factor,
        y="frequency",
        category_orders={factor: category_order},
        title=f"Claim frequency by {factor_label.lower()}",
        labels={factor: factor_label, "frequency": "Claims per policy-year"},
    )
    severity_chart = px.bar(
        experience,
        x=factor,
        y="severity_reported",
        category_orders={factor: category_order},
        title=f"{loss_basis} reported severity by {factor_label.lower()}",
        labels={factor: factor_label, "severity_reported": "Average claim amount (€)"},
    )
    frequency_chart.update_yaxes(tickformat=".3f")
    severity_chart.update_yaxes(tickprefix="€", separatethousands=True)
    chart1.plotly_chart(frequency_chart, width="stretch")
    chart2.plotly_chart(severity_chart, width="stretch")

    st.caption(
        "Frequency and severity are calculated from aggregated numerators and "
        "denominators, not by averaging policy-level ratios."
    )
    with st.expander("View underlying experience table"):
        st.dataframe(experience, width="stretch", hide_index=True)


def render_model_view() -> None:
    """Render technical model validation evidence."""
    st.header("Model performance")
    st.write(
        "XGBoost is the predictive candidate; the Poisson GLM remains the "
        "transparent actuarial benchmark."
    )

    frequency = load_csv("frequency_model_metrics.csv")
    frequency["display_model"] = model_names(frequency["model"])
    selected = frequency.loc[frequency["model"] == "xgboost"].iloc[0]
    benchmark = frequency.loc[frequency["model"] == "poisson_glm"].iloc[0]

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Selected model", "XGBoost")
    metric2.metric(
        "Test Poisson deviance",
        f"{selected['poisson_deviance']:.4f}",
        delta=f"{selected['poisson_deviance'] - benchmark['poisson_deviance']:.4f} vs GLM",
        delta_color="inverse",
    )
    metric3.metric("Normalized Gini", f"{selected['normalized_gini']:.4f}")

    left, right = st.columns(2)
    deviance_chart = px.bar(
        frequency,
        x="display_model",
        y="poisson_deviance",
        title="Poisson deviance — lower is better",
        labels={"display_model": "Model", "poisson_deviance": "Poisson deviance"},
    )
    gini_chart = px.bar(
        frequency,
        x="display_model",
        y="normalized_gini",
        title="Normalized Gini — higher is better",
        labels={"display_model": "Model", "normalized_gini": "Normalized Gini"},
    )
    left.plotly_chart(deviance_chart, width="stretch")
    right.plotly_chart(gini_chart, width="stretch")

    st.subheader("Pure-premium performance")
    pure_premium = load_csv("pure_premium_model_metrics.csv")
    pure_premium["display_model"] = model_names(pure_premium["model"])
    st.dataframe(
        pure_premium[
            ["display_model", "tweedie_deviance", "actual_to_expected"]
        ].rename(columns={"display_model": "model"}),
        width="stretch",
        hide_index=True,
        column_config={
            "tweedie_deviance": st.column_config.NumberColumn(format="%.4f"),
            "actual_to_expected": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.subheader("Actual versus predicted pure premium")
    calibration = load_csv("pure_premium_xgboost_calibration.csv")
    calibration_long = calibration.melt(
        id_vars="decile",
        value_vars=[
            "actual_pure_premium_rate",
            "predicted_pure_premium_rate",
        ],
        var_name="series",
        value_name="pure_premium_rate",
    )
    calibration_long["series"] = calibration_long["series"].map(
        {
            "actual_pure_premium_rate": "Actual",
            "predicted_pure_premium_rate": "Predicted",
        }
    )
    calibration_chart = px.line(
        calibration_long,
        x="decile",
        y="pure_premium_rate",
        color="series",
        markers=True,
        title="Pure-premium calibration by predicted-risk decile",
        labels={
            "decile": "Predicted-risk decile",
            "pure_premium_rate": "Pure premium per policy-year (€)",
            "series": "",
        },
    )
    calibration_chart.update_yaxes(tickprefix="€", separatethousands=True)
    st.plotly_chart(calibration_chart, width="stretch")

    chart_column, table_column = st.columns([3, 2])
    importance = load_csv("frequency_xgboost_importance.csv").head(15)
    importance = importance.sort_values("importance")
    importance["feature"] = (
        importance["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    importance_chart = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Top XGBoost feature importances",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    chart_column.plotly_chart(importance_chart, width="stretch")

    with table_column:
        st.subheader("Five-fold validation")
        cross_validation = load_csv("frequency_cross_validation_summary.csv")
        cross_validation["model"] = model_names(cross_validation["model"])
        st.dataframe(cross_validation, width="stretch", hide_index=True)

    with st.expander("Severity-model results"):
        severity = load_csv("severity_model_metrics.csv")
        st.dataframe(severity, width="stretch", hide_index=True)
        st.write(
            "Capping stabilised severity, but the Gamma GLM did not beat its "
            "constant-severity baseline. Pricing therefore uses capped portfolio "
            "severity with the XGBoost frequency model."
        )

    st.info(
        "Feature importance measures how much XGBoost used a variable; it does "
        "not show whether that variable increases or decreases risk."
    )


def main() -> None:
    """Render the complete Streamlit application."""
    config = load_config()

    st.title("Motor Insurance Pricing Platform")
    st.caption(
        "Experience analysis, predictive modelling and constructed technical "
        "premiums for the freMTPL2 portfolio."
    )

    executive_tab, risk_tab, model_tab = st.tabs(
        ["Executive", "Risk analysis", "Model performance"]
    )
    with executive_tab:
        render_executive_view(config)
    with risk_tab:
        render_risk_view()
    with model_tab:
        render_model_view()

    st.divider()
    st.caption(
        "Portfolio results are analytical estimates from the held-out test sample. "
        "They are not quotes or observed policy premiums."
    )


if __name__ == "__main__":
    main()
