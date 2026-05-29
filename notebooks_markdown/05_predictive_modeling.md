# Predictive Modeling

This notebook asks whether player value can be predicted one season ahead. The target is `next_value_score`, which is next season's position-adjusted value score.

I treat this as a forecasting problem, so the main concern is not fitting the past as closely as possible. The important question is whether information available in one season helps predict future value in later seasons.


## Load Cleaned Data and Shared Pipeline Helpers

The modeling notebook uses the same value-score helper as Notebook 03. I do this so the modeling dataset uses the same traded-player aggregation, games-played filter, and scoring definition as the rest of the project.



```python
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")


def find_project_root(expected_file="data/processed/skill_player_seasons_2016_2025.csv"):
    """Find the repo root from common VS Code/Jupyter working directories."""
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / expected_file).exists():
            return candidate
    raise FileNotFoundError(
        "Could not find " + expected_file + " from working directory " + str(Path.cwd())
    )


project_root = find_project_root()
processed_dir = project_root / "data" / "processed"
figures_dir = project_root / "outputs" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(project_root / "src"))
from prediction_report import (
    add_player_history_features,
    create_next_season_targets,
    create_player_season_value_scores,
)

skill_seasons_path = processed_dir / "skill_player_seasons_2016_2025.csv"
skill_player_team = pd.read_csv(skill_seasons_path)

value_scored = create_player_season_value_scores(skill_player_team)
player_season = add_player_history_features(value_scored)
player_season = create_next_season_targets(player_season)

target = "next_value_score"
modeling_df = player_season.dropna(subset=[target]).copy()

print("Raw player-season-team rows:", skill_player_team.shape)
print("Value-scored player-season rows:", value_scored.shape)
print("Rows with next-season target:", modeling_df.shape)

value_scored.head()

```

## Modeling Unit and Target Check

Each modeling row should represent one player-season. The next-season target is only created when the player's next qualifying row is exactly the next calendar season.

That rule prevents the model from accidentally linking across missed seasons, which would make the target less realistic.



```python
print("Duplicate player-season rows:", player_season.duplicated(["season", "player_id"]).sum())
print("Target seasons represented:", sorted(modeling_df["next_season"].dropna().astype(int).unique()))

display(
    player_season[[
        "season", "player_display_name", "position", "team", "games_played",
        "value_score", "value_score_prev", "value_score_last2_avg",
        "value_score_last3_avg", "value_score_trend_2yr",
        "next_season", "next_value_score"
    ]].head(12)
)

```

## Feature Overlap Check

This heatmap helps explain why I do not just put every possible feature into one model and call it done. Some variables are different versions of the same signal, such as raw EPA, standardized EPA, and EPA per game.

Checking overlap makes the feature sets easier to defend and helps avoid building a model that looks complex but is mostly repeating the same information.



```python
overlap_cols = [
    "value_epa_total", "value_epa_per_game", "value_score", "value_score_per_game",
    "games_played", "yards_per_game", "tds_per_game",
    "scrimmage_yards_per_game", "scrimmage_touches_per_game",
    "yards_per_scrimmage_touch", "qb_yards_per_play", "interceptions_per_game"
]

overlap_cols = [col for col in overlap_cols if col in modeling_df.columns]

plt.figure(figsize=(11, 8))
sns.heatmap(modeling_df[overlap_cols].corr(numeric_only=True), cmap="vlag", center=0, linewidths=0.5)
plt.title("Correlation Among Candidate Predictive Features")
plt.show()

```

## Train, Validation, and Test Split

The split is time-aware because the project is trying to predict the future from the past.

- Train: 2016-2022 seasons
- Validation: 2023 season
- Test: 2024 season

The 2024 rows predict 2025 value, so 2025 is used only as the outcome for testing, not as information the model could have known in 2024.



```python
train_df = modeling_df[modeling_df["season"].between(2016, 2022)].copy()
valid_df = modeling_df[modeling_df["season"].eq(2023)].copy()
development_df = modeling_df[modeling_df["season"].between(2016, 2023)].copy()
test_df = modeling_df[modeling_df["season"].eq(2024)].copy()

target = "next_value_score"

print("Train:", train_df.shape, train_df["season"].min(), train_df["season"].max())
print("Validation:", valid_df.shape, valid_df["season"].unique())
print("Development for final fit:", development_df.shape, development_df["season"].min(), development_df["season"].max())
print("Test:", test_df.shape, test_df["season"].unique())

```

## Define Feature Sets

Each feature set answers a slightly different question.

- `profile`: player context and availability only.
- `raw_production`: current-season raw production and EPA.
- `standardized_value`: current-season value scores.
- `usage_efficiency`: role and efficiency signals without directly using the main value score.
- `enhanced_history`: current-season production plus prior value, rolling averages, trends, and recent games played.

The enhanced-history set is the main modeling upgrade because it tests whether multi-year context improves next-season prediction.



```python
profile_features = [
    "position", "age", "years_exp", "draft_number", "games_played"
]

raw_production_features = [
    "position", "age", "years_exp", "draft_number", "games_played",
    "value_epa_total", "value_epa_per_game", "yards_per_game", "tds_per_game"
]

history_features = [
    "prior_qualifying_seasons", "value_score_prev", "value_score_last2_avg",
    "value_score_last3_avg", "value_score_trend_2yr", "value_epa_total_prev",
    "value_epa_per_game_prev", "games_played_prev", "games_played_last2_sum",
    "games_played_last3_avg", "yards_per_game_prev", "tds_per_game_prev"
]

feature_sets = {
    "profile": profile_features,
    "raw_production": raw_production_features,
    "standardized_value": [
        "position", "age", "years_exp", "draft_number", "games_played",
        "value_score", "value_score_per_game", "value_score_gap"
    ],
    "usage_efficiency": [
        "position", "age", "years_exp", "draft_number", "games_played",
        "attempts", "carries", "targets", "receptions",
        "scrimmage_touches_per_game", "yards_per_scrimmage_touch",
        "scrimmage_tds_per_game", "qb_yards_per_play", "qb_tds_per_game",
        "interceptions_per_game"
    ],
    "enhanced_history": raw_production_features + history_features,
}

feature_sets = {
    name: [col for col in cols if col in modeling_df.columns]
    for name, cols in feature_sets.items()
}

pd.DataFrame({
    "feature_set": list(feature_sets.keys()),
    "feature_count": [len(cols) for cols in feature_sets.values()],
    "features": [", ".join(cols) for cols in feature_sets.values()]
})

```

## Model Setup

I start with a dummy model so there is a real baseline. If the actual models cannot beat a simple average prediction, then the project should not claim predictive value.

The other models cover a few reasonable approaches: linear regression, Ridge regression, Random Forest, and Gradient Boosting. Linear models are easier to interpret, while tree-based models can capture nonlinear patterns.



```python
models = {
    "dummy_mean": DummyRegressor(strategy="mean"),
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=10.0),
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingRegressor(random_state=42)
}


def make_pipeline(feature_cols, model):
    categorical_cols = [col for col in feature_cols if col == "position"]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def evaluate_predictions(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }

```

## Train and Compare Models

This table compares model and feature-set combinations using validation and test metrics. I use validation performance for model selection and keep the test set as a final check.

This separation matters because choosing the model based on the test set would make the test result less honest.



```python
results = []
fitted_models = {}

for feature_set_name, feature_cols in feature_sets.items():
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    for model_name, model in models.items():
        if model_name == "dummy_mean" and feature_set_name != "profile":
            continue

        pipeline = make_pipeline(feature_cols, clone(model))
        pipeline.fit(X_train, y_train)

        valid_pred = pipeline.predict(X_valid)
        test_pred = pipeline.predict(X_test)

        valid_metrics = evaluate_predictions(y_valid, valid_pred)
        test_metrics = evaluate_predictions(y_test, test_pred)

        results.append({
            "feature_set": feature_set_name,
            "model": model_name,
            "n_features": len(feature_cols),
            "valid_mae": valid_metrics["mae"],
            "valid_rmse": valid_metrics["rmse"],
            "valid_r2": valid_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"]
        })

        fitted_models[(feature_set_name, model_name)] = pipeline

results_df = pd.DataFrame(results).sort_values(["valid_rmse", "test_rmse"])
results_df

```

## Visualize Model Results

This chart gives a quicker read on the model comparison table. Lower RMSE is better.

I am looking for two things: whether any model beats the dummy baseline, and whether the added feature complexity is actually helping enough to justify itself.



```python
plt.figure(figsize=(12, 6))
plot_results = results_df.sort_values("valid_rmse")
sns.barplot(data=plot_results, x="valid_rmse", y="feature_set", hue="model")
plt.title("Validation RMSE by Model and Feature Set")
plt.xlabel("Validation RMSE: next-season value score")
plt.ylabel("Feature set")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

```

## Hyperparameter Tuning With Rolling Validation

A single validation season can be misleading in football because one season may be unusual. To make the tuning less dependent on 2023 alone, I use rolling validation folds.

Each fold trains on earlier seasons and validates on the next season. This better matches the real forecasting problem, where future seasons are not randomly mixed with past seasons.



```python
rolling_validation_years = [2020, 2021, 2022, 2023]
rolling_folds = []

for valid_year in rolling_validation_years:
    fold_train = modeling_df[modeling_df["season"].lt(valid_year)].copy()
    fold_valid = modeling_df[modeling_df["season"].eq(valid_year)].copy()

    if not fold_train.empty and not fold_valid.empty:
        rolling_folds.append({
            "valid_year": valid_year,
            "train": fold_train,
            "valid": fold_valid
        })

pd.DataFrame({
    "valid_year": [fold["valid_year"] for fold in rolling_folds],
    "train_rows": [len(fold["train"]) for fold in rolling_folds],
    "valid_rows": [len(fold["valid"]) for fold in rolling_folds],
    "train_start": [fold["train"]["season"].min() for fold in rolling_folds],
    "train_end": [fold["train"]["season"].max() for fold in rolling_folds]
})

```


```python
tuning_models = {
    "ridge": Ridge(),
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(random_state=42)
}

tuning_grids = {
    "ridge": {
        "alpha": [0.1, 1.0, 10.0, 50.0, 100.0]
    },
    "random_forest": {
        "n_estimators": [300],
        "max_depth": [5, 7, None],
        "min_samples_leaf": [10, 20],
        "max_features": [0.5, 0.75]
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.05],
        "max_depth": [2, 3],
        "min_samples_leaf": [10, 20],
        "subsample": [0.8]
    }
}

# Tuning focuses on the production feature set currently used by the report and
# the enhanced-history version that adds prior seasons. This directly answers
# whether the multi-year feature upgrade helps after tuning.
tuning_feature_sets = [
    "raw_production",
    "enhanced_history",
]

tuning_records = []

for feature_set_name in tuning_feature_sets:
    feature_cols = feature_sets[feature_set_name]

    for model_name, base_model in tuning_models.items():
        for params in ParameterGrid(tuning_grids[model_name]):
            params_json = json.dumps(params, sort_keys=True)

            for fold in rolling_folds:
                fold_train = fold["train"]
                fold_valid = fold["valid"]

                model = clone(base_model).set_params(**params)
                pipeline = make_pipeline(feature_cols, model)
                pipeline.fit(fold_train[feature_cols], fold_train[target])

                valid_pred = pipeline.predict(fold_valid[feature_cols])
                valid_metrics = evaluate_predictions(fold_valid[target], valid_pred)

                tuning_records.append({
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "params_json": params_json,
                    "valid_year": fold["valid_year"],
                    "valid_mae": valid_metrics["mae"],
                    "valid_rmse": valid_metrics["rmse"],
                    "valid_r2": valid_metrics["r2"]
                })

tuning_fold_results_df = pd.DataFrame(tuning_records)

tuning_results_df = (
    tuning_fold_results_df
    .groupby(["feature_set", "model", "params_json"], as_index=False)
    .agg(
        mean_valid_mae=("valid_mae", "mean"),
        mean_valid_rmse=("valid_rmse", "mean"),
        std_valid_rmse=("valid_rmse", "std"),
        mean_valid_r2=("valid_r2", "mean")
    )
    .sort_values(["mean_valid_rmse", "std_valid_rmse"])
)

tuning_results_df.head(15)

```

## Tuned Candidate Comparison

The table above gives the exact parameter settings, while this chart makes the leading candidates easier to compare.

I am not expecting tuning to magically solve sports forecasting. The goal is to make the final model choice more disciplined.



```python
top_tuned = tuning_results_df.head(12).copy()
top_tuned["candidate"] = (
    top_tuned["model"] + " | " + top_tuned["feature_set"]
)

plt.figure(figsize=(11, 6))
sns.barplot(
    data=top_tuned,
    x="mean_valid_rmse",
    y="candidate",
    hue="model",
    dodge=False
)
plt.title("Top Tuned Candidates by Rolling-Validation RMSE")
plt.xlabel("Mean rolling-validation RMSE")
plt.ylabel("Model and feature set")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

```

## Tuned Model Selection

The selected tuned model uses a small simplicity rule. First, candidates are ranked by average rolling-validation RMSE. If several models are within 0.002 RMSE of the best result, I prefer the more regularized Random Forest: depth-limited trees, fewer features per split, and a larger minimum leaf size.

This is a judgment call, but it is intentional. A tiny RMSE difference is not enough reason to choose a less constrained model when the simpler version is easier to defend.



```python
best_baseline_row = results_df.iloc[0]

SIMPLICITY_RMSE_TOLERANCE = 0.002

tuned_candidates = tuning_results_df.copy()
tuned_candidates["params"] = tuned_candidates["params_json"].apply(json.loads)
tuned_candidates["max_depth"] = tuned_candidates["params"].apply(
    lambda params: params.get("max_depth") if isinstance(params, dict) else np.nan
)
tuned_candidates["uses_unbounded_depth"] = tuned_candidates["max_depth"].isna()
tuned_candidates["min_samples_leaf"] = tuned_candidates["params"].apply(
    lambda params: params.get("min_samples_leaf", 0) if isinstance(params, dict) else 0
)
tuned_candidates["max_features"] = tuned_candidates["params"].apply(
    lambda params: params.get("max_features", np.nan) if isinstance(params, dict) else np.nan
)

best_rmse = tuned_candidates["mean_valid_rmse"].min()
near_best_candidates = tuned_candidates[
    tuned_candidates["mean_valid_rmse"] <= best_rmse + SIMPLICITY_RMSE_TOLERANCE
].copy()

# Prefer a more regularized candidate when validation performance is effectively tied.
# This makes the final model easier to defend and reduces overfitting risk.
best_tuned_row = (
    near_best_candidates
    .sort_values(
        ["uses_unbounded_depth", "max_features", "min_samples_leaf", "mean_valid_rmse"],
        ascending=[True, True, False, True]
    )
    .iloc[0]
)
best_tuned_params = best_tuned_row["params"]

# Refit the best untuned baseline on the full development window so the test
# comparison uses the same training seasons as the tuned model.
baseline_key = (best_baseline_row["feature_set"], best_baseline_row["model"])
baseline_features = feature_sets[best_baseline_row["feature_set"]]
baseline_model = make_pipeline(baseline_features, clone(models[best_baseline_row["model"]]))
baseline_model.fit(development_df[baseline_features], development_df[target])
baseline_refit_pred = baseline_model.predict(test_df[baseline_features])
baseline_refit_metrics = evaluate_predictions(test_df[target], baseline_refit_pred)

best_key = (best_tuned_row["feature_set"], best_tuned_row["model"] + "_tuned")
best_features = feature_sets[best_tuned_row["feature_set"]]
best_model_template = clone(tuning_models[best_tuned_row["model"]]).set_params(**best_tuned_params)
best_model = make_pipeline(best_features, best_model_template)
best_model.fit(development_df[best_features], development_df[target])

tuned_test_pred = best_model.predict(test_df[best_features])
tuned_test_metrics = evaluate_predictions(test_df[target], tuned_test_pred)

selection_summary = pd.DataFrame([
    {
        "selection": "best_single_validation_baseline_refit",
        "feature_set": best_baseline_row["feature_set"],
        "model": best_baseline_row["model"],
        "single_validation_rmse": best_baseline_row["valid_rmse"],
        "rolling_validation_rmse": np.nan,
        "test_rmse": baseline_refit_metrics["rmse"],
        "test_r2": baseline_refit_metrics["r2"],
        "selection_rule": "best 2023 validation RMSE",
        "params": "default notebook settings"
    },
    {
        "selection": "simplicity_adjusted_rolling_validation_tuned_refit",
        "feature_set": best_tuned_row["feature_set"],
        "model": best_tuned_row["model"],
        "single_validation_rmse": np.nan,
        "rolling_validation_rmse": best_tuned_row["mean_valid_rmse"],
        "test_rmse": tuned_test_metrics["rmse"],
        "test_r2": tuned_test_metrics["r2"],
        "selection_rule": f"within {SIMPLICITY_RMSE_TOLERANCE} RMSE of best rolling model; prefer depth-limited and higher-leaf forest",
        "params": best_tuned_params
    }
])

print("Lowest rolling-validation RMSE:", round(best_rmse, 6))
print("Selected tuned model:", best_key)
print("Selected tuned parameters:", best_tuned_params)
display(
    near_best_candidates[[
        "feature_set", "model", "params_json", "mean_valid_rmse",
        "std_valid_rmse", "max_features", "min_samples_leaf", "uses_unbounded_depth"
    ]]
    .sort_values(["uses_unbounded_depth", "max_features", "min_samples_leaf"], ascending=[True, True, False])
)
display(selection_summary)

```

## Rolling Validation Error by Position

The model is pooled across positions and includes `position` as a feature. That does not guarantee it works equally well for every position.

This table checks error separately for QBs, RBs, WRs, and TEs. If one position has much larger error or a consistent bias, that would be evidence for testing position-specific models later.



```python
selected_rolling_predictions = []

for fold in rolling_folds:
    fold_train = fold["train"]
    fold_valid = fold["valid"]

    model = clone(tuning_models[best_tuned_row["model"]]).set_params(**best_tuned_params)
    pipeline = make_pipeline(best_features, model)
    pipeline.fit(fold_train[best_features], fold_train[target])

    fold_pred = pipeline.predict(fold_valid[best_features])
    fold_records = fold_valid[[
        "season", "player_id", "player_display_name", "position", target
    ]].copy()
    fold_records["predicted_next_value_score"] = fold_pred
    fold_records["residual"] = fold_records[target] - fold_records["predicted_next_value_score"]
    fold_records["abs_residual"] = fold_records["residual"].abs()
    fold_records["valid_year"] = fold["valid_year"]
    selected_rolling_predictions.append(fold_records)

selected_rolling_predictions = pd.concat(selected_rolling_predictions, ignore_index=True)

position_validation = (
    selected_rolling_predictions
    .groupby("position", as_index=False)
    .agg(
        validation_rows=("player_id", "count"),
        mean_actual_next_value=(target, "mean"),
        mean_predicted_next_value=("predicted_next_value_score", "mean"),
        bias=("residual", "mean"),
        mae=("abs_residual", "mean"),
        rmse=("residual", lambda s: np.sqrt(np.mean(np.square(s))))
    )
    .sort_values("position")
)

position_validation

```

## Final Selected Model Check

This cell keeps the original one-season validation winner visible for comparison, but the final report uses the simplicity-adjusted rolling-validation model selected above.

I include this because it is useful to see how the more careful selection process differs from the first quick validation pass.



```python
# The final selected model is assigned in the tuning section above.
# This cell keeps the original one-season baseline visible for comparison.
best_baseline_row = results_df.iloc[0]
best_baseline_key = (best_baseline_row["feature_set"], best_baseline_row["model"])

print("Best one-season validation baseline:", best_baseline_key)
display(best_baseline_row.to_frame().T)

```

## Test-Set Predictions and Residuals

Residuals show where the model was too optimistic or too pessimistic on the held-out test season.

Positive residuals mean the player outperformed the model's prediction. Negative residuals mean the player underperformed. These examples are useful because they show the kinds of player situations the model struggles with.



```python
test_predictions = test_df.copy()
test_predictions["predicted_next_value_score"] = best_model.predict(test_df[best_features])
test_predictions["prediction_residual"] = (
    test_predictions["next_value_score"] - test_predictions["predicted_next_value_score"]
)

display(
    test_predictions[[
        "season", "player_display_name", "position", "team", "games_played",
        "value_score", "next_season", "next_value_score",
        "predicted_next_value_score", "prediction_residual"
    ]]
    .sort_values("prediction_residual", ascending=False)
    .head(15)
)

display(
    test_predictions[[
        "season", "player_display_name", "position", "team", "games_played",
        "value_score", "next_season", "next_value_score",
        "predicted_next_value_score", "prediction_residual"
    ]]
    .sort_values("prediction_residual", ascending=True)
    .head(15)
)

```

## Actual vs Predicted Test Performance

A perfect model would place every point on the diagonal line. I do not expect that here because NFL performance changes quickly with injuries, roles, coaching, teammates, and player development.

This chart is mainly a calibration check: are predictions generally moving in the right direction, or are they just noise?



```python
plt.figure(figsize=(7, 7))
sns.scatterplot(
    data=test_predictions,
    x="predicted_next_value_score",
    y="next_value_score",
    hue="position",
    alpha=0.75
)
min_val = min(test_predictions["predicted_next_value_score"].min(), test_predictions["next_value_score"].min())
max_val = max(test_predictions["predicted_next_value_score"].max(), test_predictions["next_value_score"].max())
plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
plt.title("Actual vs Predicted Next-Season Value Score, Test Set")
plt.xlabel("Predicted next-season value score")
plt.ylabel("Actual next-season value score")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

```

## Permutation Importance for the Selected Model

Permutation importance asks how much model performance gets worse when each feature is shuffled.

This is helpful, but it is not causal. Correlated features can split importance between each other, so I use this as a model diagnostic rather than proof that one variable causes future value.



```python
importance = permutation_importance(
    best_model,
    test_df[best_features],
    test_df[target],
    n_repeats=10,
    random_state=42,
    scoring="neg_root_mean_squared_error"
)

importance_df = (
    pd.DataFrame({
        "feature": best_features,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std
    })
    .sort_values("importance_mean", ascending=False)
)

importance_df

```


```python
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(15), x="importance_mean", y="feature")
plt.title("Permutation Importance for Selected Model")
plt.xlabel("Increase in RMSE when shuffled")
plt.ylabel("Feature")
plt.show()

```

## Modeling Takeaways

This notebook is designed to answer four questions:

1. Can current-season information predict next-season value better than a simple baseline?
2. Which feature set is useful without overloading the model with redundant signals?
3. Does rolling-validation tuning improve the model choice compared with a one-season split?
4. Does the pooled model behave differently by position?

The enhanced-history feature set is the main modeling improvement because it uses prior-year value, rolling averages, trends, and recent games played. The final Random Forest is depth-limited because the unconstrained version only won by a tiny amount, and the simpler model is easier to defend.

The results should still be interpreted carefully. Tuning does not solve injuries, role changes, depth-chart changes, scheme changes, or the general noise of NFL forecasting. The model is most useful for tiers, risk flags, and screening players for deeper review.
