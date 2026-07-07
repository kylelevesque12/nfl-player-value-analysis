# 2026 rookie class: projected fantasy production

This scores the incoming draft class with the hierarchical Bayesian
hurdle model, fit on every earlier rookie class (2016 through 2025) with no held-out year — a production fit, not a
validation run. See `report/rookie_bayes_projection.md` for the
model's out-of-sample accuracy on past classes.

## From per-game rate to a season total

The hurdle model projects two things directly: the probability a
rookie plays meaningfully (at least 4 games) and his expected
PPR-per-game rate if he does. Neither predicts how many games he
plays if he clears that bar, so the season total here multiplies the
per-game rate by a further, simple estimate: the historical average
games played among rookies at his position who cleared the hurdle.
This is a documented approximation, not a player-specific games
model — a rookie who wins a starting job in Week 1 and one who
starts Week 9 get the same games multiplier if they share a
position.

Expected games played, if the hurdle is cleared, by position:

| Position | Expected games |
| --- | ---: |
| QB | 10.3 |
| RB | 10.8 |
| TE | 10.1 |
| WR | 11.1 |

## Top projected rookies

| Player | Pos | Pick | P(plays) | Proj PPR | 80% range |
| --- | --- | ---: | ---: | ---: | ---: |
| Jeremiyah Love | RB | 3 | 100% | 206 | 148-264 |
| Carnell Tate | WR | 4 | 100% | 157 | 107-209 |
| Jordyn Tyson | WR | 8 | 100% | 134 | 84-184 |
| Jadarian Price | RB | 32 | 100% | 125 | 68-180 |
| KC Concepcion | WR | 24 | 100% | 109 | 58-160 |
| Makai Lemon | WR | 20 | 100% | 106 | 55-157 |
| Denzel Boston | WR | 39 | 100% | 101 | 50-151 |
| Omar Cooper Jr. | WR | 30 | 100% | 98 | 45-151 |
| Kenyon Sadiq | TE | 16 | 100% | 96 | 62-129 |
| De'Zhaun Stribling | WR | 33 | 100% | 91 | 41-142 |
| Germie Bernard | WR | 47 | 100% | 89 | 38-139 |
| Caleb Douglas | WR | 75 | 99% | 83 | 33-131 |
| Chris Brazzell II | WR | 83 | 98% | 81 | 32-131 |
| Ja'Kobi Lane | WR | 80 | 98% | 81 | 31-131 |
| Antonio Williams | WR | 71 | 99% | 80 | 29-130 |

## Limitations

- Bio data (height, weight) for this class comes from pre-draft
  combine testing joined by name, not a shared ID (nflverse hasn't
  assigned this class stable cross-source IDs yet); the match rate
  is reported at fetch time by `scripts/fetch_rookie_class.py`.
- Age at draft is taken directly from the draft-picks feed (no
  birth_date is published for a class this new); computed the same
  way (age at the ~April draft) as the historical feature, so the
  two sources are consistent.
- The games-played conversion is a position-level historical
  average, not a player-specific projection; see above.
- Incumbent-context features (established starter, recent
  extension, prior-year starting QB production) use the 2025 season, the same leakage-safe convention as the
  historical training data.
