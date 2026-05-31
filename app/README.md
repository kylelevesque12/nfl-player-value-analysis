# NFL Player Value Dashboard

This Streamlit app turns the project output tables into an interactive dashboard with three product views: front office, fantasy football, and weekly win projection.

## Run Locally

From the project root:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The dashboard reads committed output tables from `outputs/tables/`.

## Sections

- Front Office Perspective: value projections, player lookup, salary efficiency, and validation
- Fantasy Football Perspective: 2026 PPR fantasy projections and validation
- Weekly Win Projection: rolling backtest game probabilities and validation
- Methodology And Reports: project checks and written summaries

## Rebuild Data

Before running the app after major project changes:

```bash
python scripts/run_pipeline.py
```

The app does not train models directly. It presents the reproducible outputs
created by the pipeline.
