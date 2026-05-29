# NFL Player Value Dashboard

This Streamlit app turns the project output tables into an interactive dashboard.

## Run Locally

From the project root:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The dashboard reads committed output tables from `outputs/tables/`.

## Pages

- Overview
- 2026 Player Predictions
- Player Lookup
- Salary Efficiency
- Model Validation
- Methodology
- Reports

## Rebuild Data

Before running the app after major project changes:

```bash
python scripts/run_pipeline.py
```

The app does not train models directly. It presents the reproducible outputs
created by the pipeline.
