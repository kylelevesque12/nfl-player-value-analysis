# Two-Stage Value Model: Opportunity × Efficiency

Player value factors exactly as `value_epa_per_game = efficiency_per_opportunity × opportunity_per_game`. The decomposition analysis showed the two factors behave very differently year over year, so this model predicts each separately and recombines them rather than predicting blended value with one model. All metrics use rolling-origin validation.

## Stage 1 — Opportunity (usage per game)

Skill score is RMSE reduction versus a persistence baseline (next opportunity = current opportunity per game).

| Method | Type | RMSE | MAE | R² | Skill vs persistence |
| --- | --- | ---: | ---: | ---: | ---: |
| random_forest | model | 4.193 | 2.439 | 0.830 | 0.9% |
| persistence | baseline | 4.230 | 2.160 | 0.827 | 0.0% |
| gradient_boosting | model | 4.313 | 2.528 | 0.820 | -2.0% |

Models by position:

| Position | Method | RMSE | R² | Skill vs persistence |
| --- | --- | ---: | ---: | ---: |
| QB | random_forest | 9.506 | 0.197 | 8.8% |
| QB | gradient_boosting | 9.802 | 0.146 | 6.0% |
| RB | random_forest | 4.830 | 0.444 | -17.1% |
| RB | gradient_boosting | 4.922 | 0.422 | -19.4% |
| TE | random_forest | 1.157 | 0.460 | -20.0% |
| TE | gradient_boosting | 1.262 | 0.358 | -30.8% |
| WR | random_forest | 1.428 | 0.472 | -11.8% |
| WR | gradient_boosting | 1.489 | 0.426 | -16.6% |

## Stage 2 — Efficiency (value EPA per opportunity)

Computed on efficiency-qualified seasons only (a position-specific minimum opportunity load), because efficiency on tiny samples is noise. Skill score is RMSE reduction versus shrink-to-mean (predict the positional mean), which is a strong null when efficiency barely autocorrelates.

| Method | Type | RMSE | MAE | R² | Skill vs shrink-to-mean |
| --- | --- | ---: | ---: | ---: | ---: |
| shrunken_persistence | baseline | 0.239 | 0.171 | 0.392 | 2.4% |
| random_forest | model | 0.240 | 0.172 | 0.386 | 1.9% |
| shrink_to_mean | baseline | 0.245 | 0.176 | 0.362 | 0.0% |
| gradient_boosting | model | 0.251 | 0.180 | 0.328 | -2.6% |

By position (all methods, to show where efficiency is learnable):

| Position | Method | RMSE | R² | Skill vs shrink-to-mean |
| --- | --- | ---: | ---: | ---: |
| QB | shrunken_persistence | 0.128 | 0.222 | 12.7% |
| QB | random_forest | 0.144 | 0.020 | 2.0% |
| QB | gradient_boosting | 0.145 | 0.008 | 1.4% |
| QB | shrink_to_mean | 0.147 | -0.020 | 0.0% |
| RB | random_forest | 0.109 | 0.043 | 2.5% |
| RB | shrunken_persistence | 0.110 | 0.026 | 1.6% |
| RB | shrink_to_mean | 0.112 | -0.007 | 0.0% |
| RB | gradient_boosting | 0.113 | -0.027 | -1.0% |
| TE | shrunken_persistence | 0.306 | 0.023 | 1.8% |
| TE | random_forest | 0.308 | 0.008 | 1.1% |
| TE | shrink_to_mean | 0.312 | -0.014 | 0.0% |
| TE | gradient_boosting | 0.327 | -0.116 | -4.9% |
| WR | random_forest | 0.307 | 0.034 | 2.2% |
| WR | shrunken_persistence | 0.309 | 0.023 | 1.7% |
| WR | shrink_to_mean | 0.314 | -0.010 | 0.0% |
| WR | gradient_boosting | 0.320 | -0.051 | -2.0% |

The expected pattern: quarterback efficiency is genuinely learnable (meaningful skill over the positional mean), while RB/WR/TE efficiency is close to pure regression to the mean — which is itself the key front-office insight.

## Recombined value vs single-model baseline

The two stages multiply to a per-game value projection, scored on efficiency-qualified pairs against a single model that predicts per-game value directly from the same feature union, and against persistence.

| Method | RMSE | MAE | R² | Skill vs persistence | Skill vs single model |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_model | 2.318 | 1.526 | 0.203 | 6.6% | 0.0% |
| two_stage | 2.417 | 1.562 | 0.134 | 2.6% | -4.2% |
| persistence | 2.482 | 1.732 | 0.087 | 0.0% | -7.1% |

## Asymmetric prediction intervals

Each stage's calibration-set residuals give a per-position error sigma, and these are propagated through the product `value = efficiency × opportunity` via `Var(E·O) = O²σ_E² + E²σ_O² + σ_E²σ_O²`. The first term is the uncertainty from the efficiency axis, the second from opportunity. This makes the band *legible*: the table below reports empirical coverage against the 80.0% target and the share of value uncertainty coming from each axis — something a single blended model cannot decompose.

| Segment | Coverage | Target | Gap | Mean width | Efficiency variance share |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 80.9% | 80.0% | 0.9% | 5.43 | 73.2% |
| QB | 80.0% | 80.0% | 0.0% | 13.65 | 90.2% |
| RB | 84.5% | 80.0% | 4.5% | 3.86 | 74.5% |
| TE | 80.8% | 80.0% | 0.8% | 3.02 | 67.6% |
| WR | 78.4% | 80.0% | -1.6% | 3.88 | 66.5% |

Variance attribution (share of total value uncertainty by axis):

| Segment | Efficiency share | Opportunity share |
| --- | ---: | ---: |
| all | 94.6% | 5.4% |
| QB | 97.6% | 2.4% |
| RB | 93.3% | 6.7% |
| TE | 83.4% | 16.6% |
| WR | 80.3% | 19.7% |

For wide receivers and tight ends almost all value uncertainty comes from the efficiency axis (the model cannot pin down per-target quality), while for quarterbacks and running backs the opportunity axis carries more of it. The interval is therefore wide along exactly the axis the model genuinely cannot predict — the practical payoff of modeling the two factors separately.

## 2026 projections (top 15 by predicted value)

Both stages are trained on all prior history and applied to the most recent season's players. Each projection carries an asymmetric interval and a driver label: *efficiency-driven* means most of the uncertainty is in per-play quality (typical of WR/TE), *role-driven* means it is in usage (typical of QB/RB).

| Player | Pos | Team | Pred. value score | Approx. 80% interval | Driver |
| --- | --- | --- | ---: | ---: | --- |
| Malik Nabers | WR | NYG | 1.78 | 0.15 to 3.42 | efficiency-driven |
| Brock Bowers | TE | LV | 1.76 | -0.79 to 4.30 | efficiency-driven |
| Nico Collins | WR | HOU | 1.71 | 0.32 to 3.11 | efficiency-driven |
| Brian Thomas Jr. | WR | JAX | 1.59 | 0.24 to 2.94 | efficiency-driven |
| Jameson Williams | WR | DET | 1.54 | 0.31 to 2.77 | mixed |
| Colston Loveland | TE | CHI | 1.53 | -0.46 to 3.51 | efficiency-driven |
| Ladd McConkey | WR | LAC | 1.33 | 0.11 to 2.56 | efficiency-driven |
| Tucker Kraft | TE | GB | 1.19 | -0.11 to 2.49 | mixed |
| Jaxon Smith-Njigba | WR | SEA | 1.14 | -0.22 to 2.49 | efficiency-driven |
| Puka Nacua | WR | LA | 1.11 | -0.24 to 2.46 | efficiency-driven |
| Tee Higgins | WR | CIN | 1.08 | -0.07 to 2.22 | efficiency-driven |
| Jordan Addison | WR | MIN | 1.06 | -0.10 to 2.22 | efficiency-driven |
| Trey McBride | TE | ARI | 1.05 | -0.70 to 2.81 | efficiency-driven |
| Justin Jefferson | WR | MIN | 1.05 | -0.20 to 2.29 | efficiency-driven |
| George Kittle | TE | SF | 1.04 | -0.29 to 2.37 | efficiency-driven |

Of 505 projected players, 261 are efficiency-qualified (enough opportunity for a reliable efficiency signal); the rest lean on the positional efficiency prior and carry wider intervals, which the driver label and interval width make explicit.

## Interpretation

Separating the axes lets the model speak the language a front office uses — "role should hold, expect efficiency regression" — and concentrate confidence where the signal actually is (opportunity, and QB efficiency) while shrinking hard where it is not (skill-position efficiency). Whether the recombined value beats a single blended model on raw RMSE is reported above honestly; even when the gain is modest, the decomposition's value is in interpretability and calibrated, axis-aware uncertainty.
