# Build Guide — Automated Motor Insurance Pricing & Portfolio Monitoring Platform

A complete, do-it-yourself plan for building a portfolio project that looks like something a real pricing team would use. You write all the substantive code; this guide tells you exactly **what to do, what to explore, what to learn, and how to know each part is done.**

Dataset: **freMTPL2** (French Motor Third-Party Liability) — a genuine, widely-used actuarial pricing dataset.

Why this project answers your MetLife interview gap: it forces you to demonstrate SQL, Python, an ETL pipeline, an actuarial experience study, a GLM **and** a machine-learning model, automation, an AI component, and a dashboard — and then to explain all of it to both technical and non-technical audiences.

> **How to use this guide.** Work top to bottom. Each stage has **Goal → Build → Learn → Watch out for → Done when**. Don't skip "Watch out for" — those are the details that separate a student project from something that looks professional, and they're exactly what an interviewer probes.

---

## The dataset: freMTPL2

Two tables, linked by policy id (`IDpol`):

**freMTPL2freq** — one row per policy:

| Column | Meaning |
|---|---|
| `IDpol` | Policy id (the key) |
| `ClaimNb` | Number of claims on the policy |
| `Exposure` | Fraction of the year the policy was active (0–1) |
| `Area` | Population-density area code (A–F) |
| `VehPower`, `VehAge`, `VehBrand`, `VehGas` | Vehicle characteristics |
| `DrivAge` | Driver age |
| `BonusMalus` | French no-claims coefficient (>100 = penalised) |
| `Density` | Inhabitants per km² of the driver's city |
| `Region` | French region code |

**freMTPL2sev** — one row per claim: `IDpol`, `ClaimAmount`.

Get it from OpenML: freq = data id **41214**, sev = data id **41215** (`sklearn.datasets.fetch_openml`). ~678k policies. The `download.py` in the scaffold already does this.

### Critical data caveats (read before modelling)

These are the things that trip people up — knowing them is itself an interview talking point:

1. **There is NO premium column.** You cannot compute a loss ratio directly. You must *construct* a technical premium (Stage 4). This is a feature, not a bug — it makes you learn premium loading.
2. **`ClaimNb` doesn't always match the number of rows in the severity table** for a policy. Decide and document how you reconcile them.
3. **`Exposure` can exceed 1** in some rows (data error). Cap at 1.
4. **Claim amounts are extremely skewed** with large outliers; a few are tiny/zero. Decide on capping/flooring and justify it.
5. **Some versions have duplicate `IDpol`.** De-duplicate.

---

## What's already scaffolded for you (optional head start)

To save you the boilerplate (not the learning), the repo already contains:

```
motor-pricing-platform/
├── config.yaml                # all settings (paths, loadings, model params)
├── requirements.txt           # dependencies
├── .env.example               # template for your OpenAI key (never commit .env)
├── .gitignore, LICENSE
├── .github/workflows/ci.yml   # GitHub Actions: runs lint + tests on every push
├── sql/
│   ├── schema.sql             # DDL for policies / claims / regions
│   └── exploration_queries.sql# 5 worked SQL queries + 5 for you to write
├── src/
│   ├── config.py              # loads config.yaml + .env
│   └── data/
│       ├── download.py        # fetches freMTPL2 from OpenML (DONE)
│       └── load_to_db.py      # loads CSVs into the SQL schema (DONE)
└── (empty dirs ready for your code: etl/ analysis/ pricing/ models/ reporting/ app/ tests/)
```

**Everything else — the ETL logic, experience study, GLM, ML model, AI report, dashboard, tests, automation — is yours to write.** The stages below tell you how.

### Target structure once you're done

```
src/
├── etl/        extract.py, clean.py
├── analysis/   experience_study.py
├── pricing/    technical_premium.py
├── models/     glm.py, ml.py
├── reporting/  ai_report.py, figures.py
app/            streamlit_app.py
tests/          test_experience_study.py, ...
run_pipeline.py
docs/           DATA_DICTIONARY.md, MODEL_VALIDATION.md
```

---

## Stage 0 — Environment & repo setup

**Goal:** a clean, reproducible project that runs on any machine.

**Build:**
- Create and activate a virtual environment, then `pip install -r requirements.txt`.
- `git init`, make your first commit, push to a new public GitHub repo.
- Copy `.env.example` to `.env` (leave the key blank for now).

**Learn:** virtual environments, `requirements.txt`, the basics of git (commit, push, branches).

**Done when:** a teammate could clone your repo, install, and the project structure is self-explanatory.

---

## Stage 1 — SQL database & exploration

**Goal:** load the data into a real SQL database and answer business questions with queries (not pandas).

**Build:**
- Run the scaffold's `download.py` then `load_to_db.py` to build `data/processed/portfolio.db` (SQLite) with three tables.
- Work through `sql/exploration_queries.sql`: 5 are solved, 5 are marked TODO — write those yourself.
- Answer these business questions in SQL:
  - *Portfolio mix:* how many policies, total exposure, average claim size?
  - *Risk segmentation:* which age bands / regions / vehicle types claim most often?
  - *Concentration:* what share of total claims comes from each region?

**Learn:** `SELECT`, `WHERE`, `JOIN`, `GROUP BY`, `HAVING`, `CASE WHEN`, and **window functions** (`RANK() OVER`, `SUM() OVER (PARTITION BY ...)`). This covers ~90% of internship SQL questions.

**Watch out for:** the difference between `WHERE` and `HAVING`; computing exposure-weighted rates in SQL (`SUM(claims)*1.0/SUM(exposure)`, not `AVG(rate)`).

**Done when:** you can answer a new ad-hoc question about the portfolio in SQL in under five minutes.

---

## Stage 2 — Python ETL pipeline

**Goal:** a repeatable Extract → Transform → Load flow in Python.

**Build (in `src/etl/`):**
- `extract.py` — read the analysis base table from SQL: one row per policy with exposure, claim count, **total claim amount per policy** (a `LEFT JOIN` to a grouped `claims` subquery), and all risk features.
- `clean.py` — apply the caveats above: cap exposure at 1, cap `BonusMalus`, handle `ClaimNb`/severity mismatches, treat outlier/zero claim amounts, de-duplicate. **Log how many rows each step changes.**

**Learn:** `pandas` (`merge`, `groupby`, `clip`, `assign`), `SQLAlchemy` engines, reading SQL into DataFrames, structuring code into functions.

**Watch out for:** silently dropping rows. Every cleaning decision should be logged and defensible ("I capped exposure at 1 because 0.2% of rows exceeded it, which is physically impossible").

**Done when:** `extract → clean` produces one tidy policy-level DataFrame, and you can state exactly how many rows you removed and why.

---

## Stage 3 — Experience study (the core actuarial work)

**Goal:** the classic one-way pricing analysis insurers actually run.

**Build (in `src/analysis/experience_study.py`):** a function that groups the policy table by any factor and returns:

```
frequency    = Σ claim_nb     / Σ exposure
severity     = Σ claim_amount / Σ claim_nb
pure_premium = Σ claim_amount / Σ exposure      (= frequency × severity)
loss_ratio   = Σ claim_amount / Σ premium       (premium comes in Stage 4)
```

Produce tables by **driver-age band, region, vehicle brand, vehicle gas, and area.**

**Learn:** frequency/severity/pure-premium decomposition — the backbone of general-insurance pricing.

**Watch out for (big one):** metrics must be **exposure-weighted** — sum the numerators and denominators across the group, *don't* average per-policy ratios. Averaging ratios is a classic mistake and a favourite interview trap. Also flag **low-exposure cells** as not credible (this motivates credibility theory if you want to go further).

**Done when:** your age-band table shows the expected U-shape (young and old drivers riskier), and pure premium equals frequency × severity to the cent.

---

## Stage 4 — Technical premium & profitability

**Goal:** turn cost into a *charged* premium so you can measure profitability.

**Build (in `src/pricing/technical_premium.py`):**

```
technical_premium = pure_premium_rate × exposure × (1 + expense_loading + profit_loading)
```

Use the empirical burning cost (or, later, your model's predicted pure premium) as `pure_premium_rate`. Loadings live in `config.yaml`. Then compute **loss ratio = actual claims / technical premium** by segment and rank the most/least profitable cohorts.

**Learn:** how a risk (pure) premium becomes an office/technical premium; what a loss ratio above/below 100% means.

**Watch out for:** be explicit that the premium is *constructed* (freMTPL2 has none). Say so in your README — honesty about data limitations reads as maturity.

**Done when:** you can name the three least profitable segments and quantify how far above break-even they are.

---

## Stage 5 — Pricing models (GLM + machine learning)

This is where you stand out from other actuarial students.

**5a — Poisson GLM for frequency (`src/models/glm.py`)**
- Model `ClaimNb` with a Poisson family **and a `log(exposure)` offset.**
- **The offset is the detail most students miss.** You observe counts, but each policy is only exposed for a fraction of a year, so `offset = log(exposure)` makes the model describe the claim *rate*. Without it your model is wrong.
- Interpret the coefficients as **relativities** (`exp(coef)`): "young drivers have 1.8× the base frequency."

**5b — Gamma GLM for severity** (optional but impressive): fit on policies with at least one claim, log link, weighted by claim count.

**5c — Pure premium**: combine frequency × severity, or fit a single **Tweedie GLM** directly on pure premium. Mention you know both routes.

**5d — Machine-learning benchmark (`src/models/ml.py`)**: train XGBoost with `objective="count:poisson"` and exposure as the sample weight. Optionally a Random Forest too.

**5e — Compare them properly.** This is the part to get right:
- **Don't rank models by RMSE** — it's a poor metric for skewed count/cost data. Use **Poisson deviance**, a **Gini coefficient / lift curve** (how well the model sorts risk), and a **calibration check** (predicted vs actual by decile).
- Use a proper **train/test split** and **cross-validation**.
- Show **feature importance** (GLM coefficients vs XGBoost importances / SHAP) and discuss where the GBM beats the GLM and what it costs you in interpretability.

**Learn:** GLMs (families, link functions, offsets, relativities), gradient boosting, and *actuarially correct* model evaluation.

**Watch out for:** data leakage (fit any transformations on train only); treating categoricals correctly; the interpretability/accuracy trade-off (insurers often keep GLMs for transparency and regulatory reasons — know why).

**Done when:** you have a table comparing GLM vs XGBoost on deviance + Gini, and you can argue which you'd put into production and why.

---

## Stage 6 — AI narrative report

**Goal:** auto-generate a management summary from your numbers.

**Build (in `src/reporting/ai_report.py`):**
- Assemble the key metrics into a dict / JSON (e.g. `{"segment": "drivers under 25", "loss_ratio": 1.38}`).
- Two modes: a **deterministic template** (no API key needed) and an **LLM mode** that sends the metrics to OpenAI for a polished narrative.
- Prompt design: instruct the model to **use only the supplied numbers and invent nothing.** Read the key from `OPENAI_API_KEY` (an environment variable) — never hard-code or commit it.

Example target output: *"Drivers under 25 show a loss ratio 38% above break-even; consider a premium review for this cohort."*

**Learn:** calling an LLM API, prompt design, structured-data-to-text, and the safety habit of never trusting a model to produce figures.

**Watch out for:** committing secrets (your `.gitignore` already excludes `.env`); LLM hallucinating numbers (constrain it); always ship the no-key fallback so the repo runs for anyone.

**Done when:** `generate_report(metrics)` returns a sensible paragraph with the key off, and a polished one with it on.

---

## Stage 7 — Dashboard (Streamlit)

**Goal:** an interactive view, designed for two audiences.

**Build (in `app/streamlit_app.py`)** three views:
- **Executive:** headline KPIs (total premium, total claims, overall loss ratio) — plain language, one number per card.
- **Risk:** loss ratio by age band, frequency by region, severity by vehicle type — the analyst's view.
- **Model:** GLM vs XGBoost metrics, feature importance, predicted-vs-actual.

**Learn:** Streamlit (`st.metric`, `st.dataframe`, charts), and **dual-audience design** — the exact skill from your interview feedback. Executives want the decision; actuaries want the method.

**Watch out for:** dumping every chart on one page. Curate. One message per view.

**Done when:** a non-actuarial friend understands the Executive page in 30 seconds, and an actuary finds the rigour on the Model page.

---

## Stage 8 — Automation (`run_pipeline.py`)

**Goal:** one command runs everything.

**Build:** orchestrate download → load → extract → clean → experience study → premium → train models → figures → AI report → write dashboard data. Add logging at each step. Make it **idempotent** (safe to re-run) and driven entirely by `config.yaml`.

**Learn:** pipeline orchestration, logging (not `print`), reproducibility (set the random seed from config).

**Watch out for:** hidden manual steps. The whole point is `python run_pipeline.py` does it all.

**Done when:** deleting `data/processed/` and running one command rebuilds the entire project.

---

## Stage 9 — Engineering rigor (your real differentiator)

Most students stop at a notebook. Don't. This is what makes it look production-grade:

- **Tests (`tests/`, pytest):** unit-test your experience-study metrics against hand-computed values (e.g. a tiny 3-policy DataFrame where you know the answer). Test that pure premium = frequency × severity, and that zero exposure yields NaN, not a crash.
- **CI (already wired in `.github/workflows/ci.yml`):** every push runs lint + tests; the green badge on your README is a strong signal.
- **Config, logging, type hints, docstrings** throughout.
- **Formatting/linting:** `black` and `flake8`.
- **Reproducibility:** one seed, set from config.
- **Data dictionary** (`docs/DATA_DICTIONARY.md`) and a **model card / validation note** (`docs/MODEL_VALIDATION.md`): assumptions, data quirks, metrics, limitations. Actuaries call this model governance — having it is rare for a student.

**Done when:** `pytest` passes locally and in CI, and `black --check` / `flake8` are clean.

---

## Stage 10 — Documentation & GitHub presentation

**Goal:** make the repo sell itself in 30 seconds.

**Build:**
- A strong **README**: one-line pitch, architecture diagram, screenshots/GIF of the dashboard, quickstart, and a short results section.
- Write two short sections — **"How it works"** (for actuaries) and **"Why it matters"** (for managers). Same project, two registers — practising the exact dual-audience skill.
- Pin the repo on your GitHub profile.

**Resume bullets to aim for:**
- Built an automated motor-insurance pricing platform (SQL → Python ETL → GLM + XGBoost → AI reporting → Streamlit dashboard) on the freMTPL2 dataset (~678k policies).
- Implemented an exposure-weighted experience study and a Poisson GLM with log-exposure offset; benchmarked against gradient boosting using Poisson deviance and Gini.
- Engineered a one-command, tested, CI-backed pipeline with config-driven runs and an LLM-generated management summary.

---

## How this improves on the ChatGPT plan

The ChatGPT plan is a good skeleton. These upgrades make it correct and senior:

1. **No premium column → build a technical premium.** The original assumed a premium existed; freMTPL2 has none. Constructing one teaches premium loading.
2. **Poisson GLM needs a `log(exposure)` offset.** Missing from the original; without it the frequency model is wrong.
3. **Evaluate with Poisson deviance + Gini/lift, not RMSE.** RMSE is the wrong metric for skewed count/cost data.
4. **Real data cleaning.** freMTPL2's documented quirks (exposure > 1, ClaimNb mismatches, outliers, duplicates) give you a genuine cleaning story.
5. **Engineering rigor:** tests, CI, config, logging, model card — the things that read as "professional," not "student."
6. **Dual-audience documentation baked in** — directly targeting your interview feedback.

---

## Suggested 6-week timeline

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Stages 0–1: setup + SQL | DB built, all 10 exploration queries answered |
| 2 | Stages 2–3: ETL + experience study | Clean base table + experience tables |
| 3 | Stages 4–5a/b: premium + GLMs | Technical premium + frequency/severity GLMs |
| 4 | Stages 5d/e: ML + comparison | XGBoost vs GLM on deviance/Gini |
| 5 | Stages 6–8: AI report + dashboard + automation | One-command pipeline + live dashboard |
| 6 | Stages 9–10: tests, CI, docs, polish | Green CI badge, README with screenshots |

---

## Interview talking points

When they ask "tell me about your Python experience," walk them through this and **narrate decisions and trade-offs, not just steps**:
- *"I used a Poisson GLM with a log-exposure offset for interpretability, then benchmarked XGBoost — it lifted Gini by X but I'd keep the GLM in production for transparency."*
- *"freMTPL2 has no premium, so I constructed a technical premium with explicit expense and profit loadings."*
- *"I evaluated with Poisson deviance and a lift curve rather than RMSE, because the cost distribution is heavily skewed."*

That reasoning — knowing *why*, and being able to pitch it to both an actuary and a manager — is the signal that fixes the feedback you got.

---

## Resources

- freMTPL2 GLM walkthrough (Python): https://github.com/lorentzenchr/Tutorial_freMTPL2
- scikit-learn Tweedie/pure-premium example: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
- glum GLM tutorial (Poisson/Gamma/Tweedie on this exact data): https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html
- chainladder-python (if you later add reserving): https://github.com/casact/chainladder-python
- statsmodels GLM docs: https://www.statsmodels.org/stable/glm.html
- Streamlit docs: https://docs.streamlit.io
