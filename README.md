# Motor Insurance Pricing & Portfolio Monitoring Platform

An end-to-end actuarial pricing project on the **freMTPL2** French motor dataset:
SQL database → Python ETL → experience study → GLM + machine-learning pricing
models → AI-generated management report → Streamlit dashboard, all run by one
command.

> This repo is a **learning build**. The setup/boilerplate is scaffolded; the
> substantive code is built by following **[BUILD_GUIDE.md](BUILD_GUIDE.md)**
> stage by stage. Expand this README as you go (see Stage 10).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data.download          # fetch freMTPL2 from OpenML (first run only)
# then follow BUILD_GUIDE.md to build the pipeline and run:
# python run_pipeline.py
# streamlit run app/streamlit_app.py
```

## What it demonstrates

SQL (joins, window functions) · Python ETL (pandas, SQLAlchemy) · actuarial
experience analysis (frequency / severity / pure premium / loss ratio) ·
Poisson GLM with exposure offset · XGBoost benchmark · LLM-generated reporting ·
interactive dashboard · tests + CI.

## Project structure

See **[BUILD_GUIDE.md](BUILD_GUIDE.md)** for the full structure, stage-by-stage
instructions, the actuarial/engineering nuances, and interview talking points.

## Data

freMTPL2 via OpenML (freq id 41214, sev id 41215). Note: the dataset has **no
premium column**, so a technical premium is constructed from the modelled pure
premium plus expense and profit loadings (see BUILD_GUIDE Stage 4).

## License

MIT — see [LICENSE](LICENSE).
