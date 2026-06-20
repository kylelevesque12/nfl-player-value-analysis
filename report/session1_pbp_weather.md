# Stage 1 — PBP depth-chart rank + weather features

## What changed

The weekly fantasy model gained two families of features.

**Play-by-play depth-chart rank.** nflverse dropped the numeric `list_rank`
field from its depth-chart feed around 2024, which had quietly killed the
"RB1 vs RB2" signal — one of the highest-value weekly features. The old
`depth_chart_rank` join was empirically populated on roughly 0.01% of
player-weeks across every season, so it contributed nothing. `src/pbp_features.py`
now rebuilds depth-chart rank directly from play-by-play usage (pass attempts
for QBs, rushes + targets for RBs, targets for WR/TE), ranks players within
each team-week-position, and exposes four shift(1)-safe rolling histories:
`pbp_depth_chart_rank_last1`, `pbp_depth_chart_rank_last4_avg`,
`pbp_targets_last4_avg`, `pbp_touches_last4_avg`. Coverage is 80–89% in every
season 2016–2025. The dead `_attach_depth_charts` join is no longer called.

**Weather.** `temp`, `wind`, and `roof` from `schedules_2016_2025.csv` are now
features (`game_temp`, `game_wind`, `is_indoor`). Indoor games (dome / closed
roof), which nflverse leaves null for temp and wind, are imputed to 70°F and
0 mph; outdoor rows with missing readings fall back to the league-wide outdoor
median. Coverage is 100%.

## Result — rolling-origin backtest (validation seasons 2023–2025)

Identical modeling frame and folds in both arms; only the feature list differs.

| Position | Baseline RMSE | Enhanced RMSE | Improvement |
|----------|--------------:|--------------:|------------:|
| Overall  | 6.0204 | 5.9439 | **+1.27%** |
| QB       | 7.4744 | 7.4280 | +0.62% |
| RB       | 6.0512 | 5.9762 | +1.24% |
| WR       | 6.0949 | 6.0148 | +1.31% |
| TE       | 4.8534 | 4.7550 | +2.03% |

Every position improved. RB and WR — the positions the stage targeted — both
land in the 1.2–1.3% range, meeting the success criterion. TE gains most
(+2.03%), consistent with depth-chart rank carrying extra signal where target
hierarchies are steepest.

## Leakage safety

The PBP features are `groupby(player_id).shift(1)`-safe — the current week's
play-by-play never enters the current week's feature. `tests/test_pbp_features.py`
pins this with a synthetic case where usage flips in the final week and
confirms the leak-free feature still reflects the prior ordering.

## Reproduce

```
python -m scripts.eval_session1_features
```

## Notes / latent bug fixed

`build_pbp_depth_chart_rank` had a latent bug (never triggered because the
module was never wired into the pipeline): the `groupby` handle was bound
before `_touches` was assigned via `usage.assign(...)`, which returns a fresh
frame. Switched to in-place assignment so the column is visible to the group.
