# Value Decomposition: Efficiency vs Opportunity

The headline `value_score` is the within season-position z-score of *total* EPA, which blends how good a player is per opportunity with how much they are used. This report splits value into two standardized axes — **efficiency** (value EPA per opportunity) and **opportunity** (usage per game) — and measures how repeatable each is year over year. A more repeatable signal is more likely to reflect stable ability, which is what a front office wants to isolate.

## Year-over-year persistence (lag-1 correlation)

Higher is more repeatable. Overall, then by position.

| Segment | Total value | Efficiency | Opportunity | n pairs |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.422 | 0.258 | 0.759 | 3086 |
| QB | 0.495 | 0.473 | 0.531 | 348 |
| RB | 0.207 | 0.221 | 0.782 | 784 |
| WR | 0.485 | 0.183 | 0.787 | 1263 |
| TE | 0.495 | 0.255 | 0.770 | 691 |

Overall, **opportunity** is the more year-over-year stable axis (opportunity 0.759 vs efficiency 0.258; total value 0.422). This is the central finding for talent evaluation: if opportunity is the more persistent component, then much of what total-EPA value 'predicts' year to year is really role stability, not ability. Modeling the two axes separately — a role/opportunity forecast times an efficiency forecast — should therefore be more honest and more useful than predicting blended total value, and it lets the front office ask the two distinct questions (How good? vs How used?) independently.

## How to use these columns

`efficiency_z` and `opportunity_z` are standardized within each season-position group, so a value of +1 means roughly one standard deviation above positional peers that season. The talent rate features (catch rate, yards per target, aDOT, YAC per reception, RACR, yards per carry, completion %, yards per attempt, passing aDOT, PACR) describe *how* production is earned and are the natural inputs for an efficiency-side model that aims at ability rather than volume.
