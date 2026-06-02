# Two-Stage Value Model — Stage 1: Opportunity

This is the first stage of a two-stage value model that predicts **opportunity** (usage per game) and **efficiency** (value EPA per opportunity) separately, then recombines them. Opportunity is the high-signal half — it persists strongly year over year — so it is modeled first. All metrics use rolling-origin validation, and the skill score is the percentage RMSE reduction versus a persistence baseline (next-season opportunity = this-season opportunity per game).

## Overall results

| Method | Type | RMSE | MAE | R² | Skill vs persistence |
| --- | --- | ---: | ---: | ---: | ---: |
| persistence | baseline | 4.230 | 2.160 | 0.827 | 0.0% |

## By position (models only)

| Position | Method | RMSE | R² | Skill vs persistence |
| --- | --- | ---: | ---: | ---: |

## Next step

Stage 2 will model next-season efficiency on efficiency-qualified rows using the talent rate features, with heavy shrinkage to the positional mean as the baseline to beat. The two stages then multiply (with expected games played) to reconstruct a value projection whose uncertainty is honest about which half is predictable.
