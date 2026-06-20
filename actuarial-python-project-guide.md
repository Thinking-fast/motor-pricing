# Actuarial Python Project Guide

A curated set of open-source projects and a concrete portfolio plan to rebuild your skills in Python — across **life, general insurance (P&C), and health** — and to practice **explaining your work to both actuarial and non-actuarial audiences**.

Built around your goals: strengthen the technical modelling *and* the coding *and* the communication (dashboards + presentations) that came up in your MetLife interview.

---

## Start here: the one project to build

If you only do one thing, build an **end-to-end insurance pricing model with a dashboard and two presentations**. It touches every skill you named in a single, interview-ready story.

**The project — Motor insurance pricing engine (freMTPL2 dataset)**

1. **Data** — the French Motor Third-Party Liability dataset (`freMTPL2`, ~678k policy-years). It's free, well-documented, and the de-facto teaching dataset for non-life pricing. Fetch it straight from scikit-learn (`fetch_openml`).
2. **Model** — build a classic **frequency × severity** model (Poisson GLM for claim count, Gamma GLM for claim size), then a single **Tweedie GLM** for pure premium. Then beat it with a **gradient-boosted model** (LightGBM/XGBoost) and compare.
3. **Validate like an actuary** — lift charts, Gini/ordered-loss curves, actual-vs-expected by risk factor, calibration. This is exactly the diagnostic vocabulary pricing teams expect.
4. **Dashboard** — wrap it in a **Streamlit** app: enter risk factors → get a predicted premium, plus portfolio views (loss ratio by segment, model lift, A-vs-E).
5. **Two presentations** — this is the communication rep:
   - *Technical deck (for actuaries):* model spec, link functions, GLM diagnostics, why Tweedie, where the GBM wins/loses, validation.
   - *One-pager / exec view (for non-actuaries):* what drives price, what it means for the portfolio, the business trade-off — no jargon, one chart per idea.

**Learn-from repos (don't start from a blank file):**

- [lorentzenchr/Tutorial_freMTPL2](https://github.com/lorentzenchr/Tutorial_freMTPL2) — a clean Python walkthrough of GLMs on exactly this dataset.
- [scikit-learn Tweedie example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html) — full working frequency/severity/Tweedie code.
- [glum French motor tutorial](https://glum.readthedocs.io/en/latest/tutorials/glm_french_motor_tutorial/glm_french_motor.html) — Poisson/Gamma/Tweedie with a fast production-grade GLM library.

Why this project: it's *the* job, not a toy. It forces real coding (not just a notebook), the full modelling pipeline, and — most importantly for you — making the same result legible to two very different audiences.

---

## The toolkit, by domain

All of these are open-source, actively maintained, and Python. Pick the ones that match what you want to go deep on.

### Life insurance

| Project | What it's for | Why it fits you | Link |
|---|---|---|---|
| **lifelib** | Library of life actuarial models — pricing, profit testing, valuation, cashflow projection, ALM, capital. Runs on `modelx`. | The most practical life-modelling sandbox in Python. Open a model, trace the formulas, change assumptions, see reserves move. Great for "replace the spreadsheet" stories. | [github.com/lifelib-dev/lifelib](https://github.com/lifelib-dev/lifelib) |
| **actuarialmath** | Life-contingent risk: survival models, annuities, premiums, reserves. Follows the SOA FAM-L syllabus and the Dickson–Hardy–Waters text. | Best for nailing the *theory* in code. Rebuild the textbook in Python and you'll never get caught out on fundamentals in an interview. | [github.com/terence-lim/actuarialmath](https://github.com/terence-lim/actuarialmath) |
| **pymort** | Loads SOA mortality tables (mort.soa.org) into Python objects. | The data layer for any life project — real tables, no manual entry. | [github.com/actuarialopensource/pymort](https://github.com/actuarialopensource/pymort) |

*Mini-project:* reproduce a term-life or endowment profit test in `lifelib`, drive mortality from a real `pymort` table, and present reserve sensitivity to a +/- mortality and interest shock.

### General insurance (P&C)

| Project | What it's for | Why it fits you | Link |
|---|---|---|---|
| **chainladder-python** | Reserving: triangles, link ratios, IBNR, Mack, bootstrap/stochastic models. CAS-backed, `pandas`/`scikit-learn`-style API. | The flagship open-source actuarial package (236★, CAS volunteer-maintained). Reserving is core P&C work and this is the standard Python tool. | [github.com/casact/chainladder-python](https://github.com/casact/chainladder-python) |
| **GEMAct** | Non-life (re)insurance: collective risk model, loss aggregation, stochastic reserving, copulas. Peer-reviewed (Annals of Actuarial Science). | Step up from reserving into capital/reinsurance modelling. Strong for a "range of outcomes" story. | [gem-analytics.github.io/gemact](https://gem-analytics.github.io/gemact/) |
| **aggregate** | Fast, accurate compound (frequency–severity) distributions without simulation; has its own modelling language (DecL). | For risk/capital and aggregate-loss work — elegant and genuinely fast. | [github.com/mynl/aggregate](https://github.com/mynl/aggregate) |

*Mini-project:* take a loss triangle in `chainladder-python`, fit a deterministic chain-ladder and a bootstrap model, then present the **IBNR point estimate vs. the full distribution** — "here's our best estimate, here's the risk around it" — to a mock management audience.

### Health insurance

Health has fewer dedicated libraries, so the move is to apply the pricing/claims-modelling toolkit to health data.

- **Medical Cost Personal dataset** (Kaggle, ~1.3k records: age, sex, BMI, smoker, region → charges) — the classic health-cost modelling starter. Build a GLM and a GBM to predict charges, then explain *what drives cost* (smoking, BMI) to a non-technical audience. [kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- The same **frequency/severity and GLM/Tweedie** workflow from the capstone transfers directly to health claims — reuse your pipeline on a health dataset for a second, contrasting case study.

### Cross-cutting (pricing & data science — applies to all three lines)

This is where modern actuarial hiring is heading, and it's the engine of the capstone: GLMs, Tweedie, and gradient boosting on insurance claims. The three freMTPL2 resources listed under "Start here" are your main references.

---

## The communication layer

This is the part that most directly answers the interview feedback. The skill isn't "make a chart" — it's **making the same result land for two audiences.**

**Tooling — pick one and go deep:**

- **Streamlit** — fastest path from a Python script to an interactive app. Best default for a portfolio dashboard. [streamlit.io](https://streamlit.io)
- **Plotly Dash** — more control/customisation, slightly more setup.
- **Shiny for Python** — natural if you also know R's Shiny.

**Dual-audience habit to practise on every project:**

- *For actuaries:* lead with method and validation. Show assumptions, diagnostics, and where the model breaks. Use the right terms (link function, deviance, lift, IBNR).
- *For non-actuaries:* lead with the decision. One message per slide, one chart per idea, money and risk in plain language, technical detail in an appendix. Replace "Tweedie GLM with log link" with "a model that predicts both how often and how much."

A simple forcing function: after each project, write **two README sections** — "How it works" and "Why it matters" — and record yourself explaining each in two minutes.

---

## Where to find more

- **actuarial-foss** — a curated list of free/open-source actuarial software across Python and R. Your map of the ecosystem. [github.com/genedan/actuarial-foss](https://github.com/genedan/actuarial-foss)
- **Actuarial Open Source Community** — the org behind `pymort` and others; benchmarks and validates open actuarial tools. [actuarialopensource.org](https://www.actuarialopensource.org/) · [github.com/actuarialopensource](https://github.com/actuarialopensource)

---

## A 6-week sequence

| Week | Focus | Output |
|---|---|---|
| 1 | freMTPL2 + GLM frequency/severity (capstone core) | Working notebook, baseline pure-premium model |
| 2 | Tweedie + gradient boosting + validation | Model comparison with lift/Gini/A-vs-E |
| 3 | Streamlit dashboard around the model | Interactive premium + portfolio app |
| 4 | Two presentations (technical + exec) | Two decks from the same project |
| 5 | Pick a domain deep-dive (life *or* P&C) | `lifelib` profit test **or** `chainladder` reserving study |
| 6 | Health case study + polish GitHub repos | Second case study; clean READMEs with "How it works / Why it matters" |

By the end you have one strong capstone, one domain deep-dive, one health contrast, a live dashboard, and reps explaining all of it to both audiences — a portfolio you can walk an interviewer through end to end.

---

## Turning this into the interview answer

The feedback you got is fixable, and this plan targets it directly:

- **Technical depth** → you've built GLMs and GBMs from the data up and can defend the choices.
- **Coding** → real repos with structure and a deployed app, not just notebooks.
- **Domain** → you've touched life, P&C, and health, and know which tool each uses.
- **Communication** → you can pitch the *same* model to an actuary and to a manager, and you've practised both.

When you talk about it, narrate the decision and the trade-off ("I used a GLM for interpretability, then tested a GBM and it lifted X but cost transparency"), not just the result. That's the signal interviewers are listening for.
