# The Draft Room planner: a whole-draft dynamic program

## The question

Most "best player available" tools answer one question at a time: who is the
best player left right now? That is the wrong question by itself, because a
draft pick's real cost is not the player in front of you, it is what you give
up by not taking someone else — and that cost depends on the whole rest of
your draft, not just your next pick. The Draft Room planner answers the
larger question directly: given everything known right now, what is the best
sequence of picks for every remaining slot on your starting roster?

## The opponent model

Between two of your own picks, only opponents draft (a snake draft with no
trades has no other structure). Opponents are modeled as always taking the
best remaining player by average draft position (ADP): a free, public
snapshot of real draft behavior from Fantasy Football Calculator. This is the
same simplifying assumption a static "value based drafting" calculator makes,
and it is a real approximation — actual opponents have their own team needs
and biases that pure ADP does not capture. It is stated here plainly rather
than hidden, and it is the reason this is called a "deterministic v1": a
planned Monte Carlo upgrade will sample opponent picks around ADP many times
and report how often each opening move wins, replacing this single assumption
with a distribution instead of removing it.

## Avoiding a leakage bug, honestly

The first working version of the planner made a subtle mistake worth naming
because it is exactly the kind of bug this project's methodology audit
exists to catch. It precomputed how many players of each position opponents
would take before each of the user's future picks, once, independent of the
user's own choices, and then searched over the user's position sequence
using that fixed table. The problem: if the plan's own first move takes the
best available running back, that running back is no longer on the board for
the very next opponent pick either — but the precomputed table did not know
that, so it would sometimes assign the same player to both the user and a
simulated opponent in the same hypothetical draft. On a hand-built test case
with a steep running back cliff, this produced a real double-count.

The fix was to abandon the two-phase design and simulate one shared timeline
instead: a single pool of undrafted players that both the user's hypothetical
picks and the simulated opponent picks draw from, so a player taken at any
point in a candidate plan is gone for every pick that follows in that same
plan. The corrected planner was re-verified against the same hand-worked
example (see `tests/test_draft_planner.py`), and the numbers now agree with a
careful manual calculation exactly.

## The plan itself

With the shared timeline in place, "how many players of position P have I
already drafted" is enough information to know exactly which player would be
available at any future pick of mine — I always take the best remaining
player at whichever position I choose. That makes the plan a bounded search
over sequences of position choices for the picks that fill a starting roster
(the roster's position minimums plus its flex slot; bench rounds beyond that
are not planned in per-position detail), maximizing total value over
replacement subject to those roster constraints. On the full 505-player 2026
board this search runs in under 30 milliseconds, fast enough to re-plan after
every single pick recorded live.

## What the plan surfaces

- **The recommended pick right now**, and the full sequence the plan expects
  to make with every remaining starter slot.
- **The cost of waiting one round** at each position: the gap between the
  best player available at that position now and the best expected to remain
  at your very next pick, which is the plain-number version of "how steep is
  this position's cliff from here."

## Limitations

- Opponents are modeled as ADP followers, not as teams with their own needs;
  this is the deterministic v1's central approximation, and the Monte Carlo
  upgrade is designed specifically to address it.
- The plan inherits the season projections' own uncertainty. Adjacent
  players are frequently statistically indistinguishable, so a plan that
  "wins" by a few points is closer to a coin flip than a certainty — the app
  surfaces this with tier language on the Draft Board rather than false
  precision here.
- Only starter-relevant picks are planned in detail; bench-round strategy
  (handcuffs, late-round fliers) is not modeled.
- 2026 rookies are not yet in the projection table the planner draws from,
  so they cannot be recommended even though real drafts take them early (see
  the roadmap item to fold rookie projections into the season rankings).
