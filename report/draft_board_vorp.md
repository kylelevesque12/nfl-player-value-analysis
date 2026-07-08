# The overall draft board: VORP and auction values

This report documents how the cross-position draft board is built.

## The question

Projected points are not comparable across positions: the 12th-best
quarterback and the 12th-best running back both start in a 12-team
league, but the freely available player behind each differs
enormously. The board therefore ranks players by VORP (value over
replacement player): projected points minus the replacement level at
the position.

## How replacement level is computed

The league's starting lineups are actually filled: 12 QB, 24 RB, 24
WR, and 12 TE as fixed starters, then 12 flex slots go to the best
remaining skill players one at a time. The replacement level at each
position is the projection of the best player left after every
starting slot is gone. The computed levels:

| Position | Replacement-level projected PPR |
| --- | ---: |
| QB | 227.1 |
| RB | 128.2 |
| WR | 134.6 |
| TE | 128.1 |

## The top of the board

| Overall | Player | Pos | VORP | Auction $ | ADP |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Puka Nacua | WR | 175 | $99 | — |
| 2 | Bijan Robinson | RB | 171 | $97 | — |
| 3 | Jaxon Smith-Njigba | WR | 166 | $94 | — |
| 4 | Jahmyr Gibbs | RB | 153 | $87 | — |
| 5 | Ja'Marr Chase | WR | 148 | $84 | — |
| 6 | Christian McCaffrey | RB | 141 | $80 | — |
| 7 | Amon-Ra St. Brown | WR | 132 | $75 | — |
| 8 | De'Von Achane | RB | 129 | $73 | — |
| 9 | Trey McBride | TE | 125 | $71 | — |
| 10 | Jonathan Taylor | RB | 115 | $66 | — |

## ADP match diagnostics

145 of 149 ADP
players matched to the projection table
(97.3%). However, one gap needs to
be stated plainly: the unmatched players inside the top 100 picks
are 2026 rookies, who are not in the season projection table yet.
Until the rookie class is scored (a planned item on the roadmap),
the board is honest about veterans but silent on rookies:


## Limitations

- Replacement level uses a standard 12-team lineup (QB, 2 RB, 2 WR,
  TE, one flex). Other league shapes shift the levels; the code takes
  the league shape as a parameter.
- Auction values split the league's discretionary budget in
  proportion to positive VORP. That is a defensible convention, not
  market truth.
- VORP inherits the projections' uncertainty. The board carries
  vorp_low / vorp_high from the 80% projection intervals, and
  adjacent players are often statistically indistinguishable.
