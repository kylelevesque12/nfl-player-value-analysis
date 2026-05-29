# 2026 Player Value Prediction Report

This notebook turns the modeling work into a report someone can actually use. It trains the final models on historical player-season data, uses 2025 player information as the input, and creates 2026 projections.

The goal is not to pretend the model can perfectly rank future NFL performance. The goal is to give a clear projection, show the uncertainty around it, and make the main risks visible.


## Why This Version Is Stronger

A simple next-season value model only learns from players who appear again in the next season. That creates survivorship bias because players who disappear from the dataset are easy to ignore.

This version separates the problem into two parts:

- **Availability model:** is the player likely to have a qualifying 2026 season?
- **Value model:** if he qualifies, what value score does the model expect?

I also include prediction intervals and validation tables because a useful sports model should be honest about uncertainty.



```python
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def find_project_root(expected_file="data/processed/player_value_scores_2016_2025.csv"):
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / expected_file).exists():
            return candidate
    raise FileNotFoundError(f"Could not find {expected_file} from {Path.cwd()}")

project_root = find_project_root()
sys.path.insert(0, str(project_root / "src"))

from prediction_report import build_2026_prediction_tables

project_root
```

## Generate Prediction Tables

This cell builds the report dataset, trains the final value and availability models, predicts 2026 outcomes from 2025 inputs, and saves the report tables.

The same function is used for the CSVs, JSON payload, and Excel workbook so the outputs stay consistent.



```python
outputs = build_2026_prediction_tables(project_root=project_root, save_outputs=True)

player_predictions = outputs["player_predictions"]
team_summary = outputs["team_summary"]
position_summary = outputs["position_summary"]
top_players = outputs["top_players"]
low_confidence = outputs["low_confidence"]
value_validation_by_position = outputs["value_validation_by_position"]
interval_validation = outputs["interval_validation"]
availability_metrics = outputs["availability_validation_metrics"]
model_notes = outputs["model_notes"]
output_dir = outputs["output_dir"]

print("Player predictions:", player_predictions.shape)
print("Team summary:", team_summary.shape)
print("Position summary:", position_summary.shape)
print("Value validation by position:", value_validation_by_position.shape)
print("Interval validation:", interval_validation.shape)
print("Output directory:", output_dir)

```

## Top 2026 Player Projections

`predicted_2026_value_score` is the main projection. The interval columns show an approximate central 80% range, and the qualifying probability estimates whether the player is likely to have enough 2026 data to be included again.

I read these columns together. A high projection with a wide interval or lower qualifying probability should be treated differently than a high projection with strong availability and a tighter range.



```python
display(
    player_predictions[[
        "player_display_name", "position", "primary_team_2025", "games_played_2025",
        "value_score_2025", "value_score_last2_avg", "predicted_2026_value_score",
        "prediction_interval_low", "prediction_interval_high",
        "predicted_2026_qualifying_probability", "availability_risk_level",
        "predicted_2026_value_tier", "confidence_level", "prediction_driver"
    ]].head(20)
)

```

## Value Model Performance by Position

The value model is pooled across positions, but I still need to check whether error differs by position.

If one position has much worse RMSE or consistent bias, that would be a sign that a future version should test separate models or position-specific features.



```python
display(value_validation_by_position)

plt.figure(figsize=(8, 4))
sns.barplot(
    data=value_validation_by_position,
    x="position",
    y="rmse",
    order=["QB", "RB", "WR", "TE"],
    color="#2563EB"
)
plt.title("Rolling Validation RMSE by Position: Value Model")
plt.xlabel("Position")
plt.ylabel("RMSE")
plt.show()

```

## Prediction Interval Calibration

The prediction interval is approximate. It combines historical rolling-validation error with disagreement across Random Forest trees.

The target is a central 80% interval. If historical coverage is much lower than 80%, the model is overconfident. If it is much higher, the intervals may be too wide to be useful.



```python
display(interval_validation)

plot_df = interval_validation[interval_validation["segment"].eq("position")].copy()

plt.figure(figsize=(8, 4))
sns.barplot(
    data=plot_df,
    x="segment_value",
    y="coverage_rate",
    order=["QB", "RB", "WR", "TE"],
    color="#2563EB"
)
plt.axhline(0.80, color="#DC2626", linestyle="--", label="Target coverage")
plt.ylim(0, 1)
plt.title("Approximate 80% Prediction Interval Coverage by Position")
plt.xlabel("Position")
plt.ylabel("Rolling-validation coverage")
plt.legend()
plt.show()

```

## Availability Model Performance

The availability model estimates whether a player will have a qualifying next-season row. I include this because missing the next season is a real outcome, not just a data inconvenience.

This helps keep availability risk visible instead of hiding it inside the value prediction.



```python
display(availability_metrics)

plt.figure(figsize=(8, 4))
sns.lineplot(
    data=availability_metrics,
    x="valid_year",
    y="roc_auc",
    marker="o",
    color="#2563EB"
)
plt.ylim(0.5, 1.0)
plt.title("Rolling Validation AUC: Next-Season Availability Model")
plt.xlabel("Validation season")
plt.ylabel("ROC AUC")
plt.show()
```

## Confidence and Risk Labels

I keep confidence and availability risk separate.

- `availability_risk_level` is about whether the player is likely to qualify next season.
- `confidence_level` is about how stable the projection looks.
- `prediction_driver` gives a short plain-English reason for the projection.

This makes the report easier to read because not every risk gets compressed into one score.



```python
summary_counts = (
    player_predictions
    .groupby(["availability_risk_level", "confidence_level"], as_index=False)
    .agg(players=("player_id", "count"))
)

display(summary_counts)

plt.figure(figsize=(8, 5))
sns.countplot(
    data=player_predictions,
    x="availability_risk_level",
    hue="confidence_level",
    order=["Low", "Medium", "High"],
    hue_order=["High", "Medium", "Low"],
    palette={"High": "#2563EB", "Medium": "#F59E0B", "Low": "#DC2626"}
)
plt.title("2026 Projection Confidence by Availability Risk")
plt.xlabel("Availability risk")
plt.ylabel("Players")
plt.legend(title="Confidence")
plt.show()
```

## Team and Position Views

The team field is `primary_team_2025`, meaning the team with the largest 2025 sample in the data. It should not be read as a guaranteed 2026 roster projection.

These summary views are meant for filtering and quick review, not for making team-level roster claims.



```python
display(team_summary.head(15))
display(position_summary)
```


```python
plt.figure(figsize=(8, 4))
sns.barplot(
    data=position_summary,
    x="position",
    y="avg_availability_adjusted_2026_value",
    color="#2563EB"
)
plt.axhline(0, color="black", linewidth=1)
plt.title("Average Availability-Adjusted 2026 Value by Position")
plt.xlabel("Position")
plt.ylabel("Average availability-adjusted projected value")
plt.show()
```

## Exported Files

This notebook saves the files that support the Excel report: player predictions, team and position summaries, validation tables, model notes, and the data dictionary.

The Excel workbook is the cleanest artifact for a reviewer, while the CSVs and JSON files make the report auditable.



```python
for path in sorted(output_dir.glob("2026_*")):
    print(path)
```

## Methodological Notes

This report is a screening and prioritization tool, not a guarantee. That is okay for this problem. Sports forecasting is hard, so a good report should show expected value, uncertainty, and risk rather than pretending to be exact.

This version is stronger because it uses multi-year history, a depth-limited model selected with time-aware validation, traded-player aggregation before filtering, position-level validation, prediction interval calibration, and a separate availability model.

Important limitations remain:

- It does not know future injuries, depth-chart changes, coaching changes, rookies, free agency, or 2026 team context.
- EPA-based production reflects team environment and usage, not pure individual talent.
- The model is pooled across positions, so position-level validation is a check, not a complete replacement for position-specific models.
- `predicted_2026_value_score`, qualifying probability, prediction interval, and driver notes should be interpreted together.
