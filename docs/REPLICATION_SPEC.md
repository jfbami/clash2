# Method specification

## Provenance

The transition-quality method specified below originates with Heo and Kim (2026), *When to Switch, Not Just What: Transition Quality Prediction in Clash Royale*, arXiv 2605.21868.
We reproduce their pipeline and their metrics, then extend it.
Read depth: full, 29 pages. See `RELATED_WORK.md`.

That paragraph is the whole of the attribution.
Everything after it is written as our specification, because it is what we are building.

## Design principle

**Metrics are fixed. Model choices are ours.**

We hold every definition, target and evaluation metric constant so our numbers are comparable to the published ones.
We do not hold the architecture constant.
Where a GRU, an MLP or a transformer is specified, we treat that as a starting point and swap it if something works better, then report both.

A result only counts as a replication when it is produced by the same metric on the same target.
It does not require the same network.

## Definitions

### Deck archetypes

Archetypes are regions of deck space, and they are the origin and destination of every transition.

We build ours outcome-blind, per D13, using the card co-occurrence space in `crdata/embedding.py`.
That space is validated: it recovers win conditions at 71.8 percent against a 21.2 percent baseline, having never seen a win or a loss.

The published method clusters instead on seven structural indicators with `KMeans(k=13)`, reporting Silhouette 0.54.
Only three of the seven are named, and the accompanying weights are never given, so that clustering cannot be reproduced exactly and we do not pretend otherwise.
Our archetypes are ours, and R11 sets the bar anything replacing them must clear.

### Player types

We classify players by how they change decks, using three features that already exist in `crdata/player_panel.py`:

- `switch_rate`: fraction of consecutive battles where the deck changed.
- `loss_reactivity`: `switch_after_loss` minus `switch_after_win`.
- `switch_drift`: mean Jaccard distance between consecutive decks when a change occurred.

`KMeans(k=3)` on these separates three types: players who never change deck, players who change constantly and mostly after losses, and players who hold a core deck with occasional excursions.

### The target

Raw transition quality:

    y_tq = WR_next - WR_current

Net transition effect, which is what a decision should actually optimise:

    ΔWR_net = y_tq - stay_baseline(s, u, b)

`stay_baseline(s, u, b)` is the mean `y_tq` among players who did **not** switch, matched on archetype `s`, player type `u`, and win-rate bucket `b`.
A switch counts as beneficial only when `ΔWR_net > 0`.

Subtracting the stay baseline is the point.
Players on a losing streak recover regardless of what they do, so `y_tq` alone rewards recommending a switch to anyone who is losing.

**We train every predictor on `ΔWR_net`, not on `y_tq`.**
The published pipeline trains its timing gate on `ΔWR_net` but its quality predictor on raw `y_tq`, and the quality predictor is what ranks destinations.
That leaves the ranking exposed to exactly the mean reversion the baseline exists to remove.
We run it both ways and report the difference.

### The pipeline

Three stages over a shared frozen encoder: **who** to advise, **when** to advise them, **what** to recommend.

**Player state encoder.** A sequence model over the last `K=10` battles.
Categorical inputs: archetype, win or loss, whether the deck changed, crown difference.
Continuous inputs: time since previous battle, average elixir.
Deck enters as the mean of its card embeddings.

    z_raw  = W_o [ →h_K ‖ ←h_1 ]
    z_cls  = z_raw + MasteryProj(mf)

`z_user` is an exponentially decayed average of earlier `z_cls`, decay 0.9, which keeps future battles out of the representation.

`mf` is a 7-dimensional summary the sequence model handles poorly on its own: average win rate, average switch rate, average elixir, tilt signal, win-rate trend, crown-score trend, switch concentration.
Tilt signal is the count of losses in the last 3 battles, an integer in `{0,1,2,3}`.

Pre-training runs five heads at once, weighting deck-change prediction and transition-type classification highest, with win/loss, player type and crown difference as auxiliary. The encoder is frozen afterwards.

**Who.** Players whose consistency already correlates with better outcomes are excluded and told to stay.

**When.** A binary classifier on whether this moment is one where switching beats staying.
Positive label when `ΔWR_net > 0`.
Stay samples are undersampled to match switch count in training only, never in validation or test.

**What.** Candidates are ranked by combining how likely a player is to adopt a deck with how much it is predicted to help:

    score(s') = α · norm_CB(s') + (1 - α) · tanh( TQP(s') / 0.1 )

Candidates require at least 3 observations for the same player type and origin archetype, and are dropped when predicted quality is not positive.

### The metrics

    SwitchGap = E[ ΔWR | recommended switch, actually switched ]
              - E[ ΔWR | recommended stay,   actually switched ]

`SwitchGap` is evaluated **only among players who actually switched**, so observed choices are never treated as correct.
That matters here: the players who switch most have the worst records, so any metric that rewards matching their choices rewards imitating the worst behaviour.

Alongside it:

- `Rec_TQP`: mean predicted `ΔWR` among approved transitions.
- `Prec@1`: fraction of approved transitions that actually improved.
- `Sw%`: share of moments the policy approves.
- `MAE` and direction accuracy on `ΔWR`.

## Benchmark figures

These are the published numbers we compare against, on 926,334 matches from 34,619 players.

| quantity | published |
|---|---|
| deck change AUC | 0.843 |
| player type accuracy | 69.7% |
| transition type accuracy | 81.0% |
| MAE on `ΔWR` | 0.0517 |
| direction accuracy | 76.9% against a 64.0% base rate |
| precision / recall / F1 | 64.5% / 90.7% / 0.753 |
| discrimination gap | +12.9%p, 95% CI [+12.3, +13.6] |
| full pipeline | Sw% 5.4%, SwitchGap +10.4%p, Prec@1 70.4% |

Player types, published:

| type | share | switch rate | post-loss | post-win | reactivity | win rate |
|---|---|---|---|---|---|---|
| never switches | 48.1% | ~0.00 | ~0.00 | ~0.00 | ~0.00 | 0.53-0.56 |
| switches after losses | 16.0% | 0.27 | 0.487 | 0.064 | 0.423 | 0.48-0.53 |
| holds a core deck | 35.9% | 0.10 | 0.148 | 0.066 | 0.082 | 0.51-0.55 |

Direction accuracy deserves reading carefully.
The 64.0% base rate means **roughly two thirds of deck switches leave the player worse off**, which is the more interesting number and is easy to skim past.

## Where the published method is underspecified

Three gaps we have to fill ourselves, recorded so our choices are visible.

**The archetype features.** Four of seven indicators unnamed, weights unnamed. We use our own outcome-blind space.

**The `WR_next` window.** `WR_current` follows from `K=10`. The length of the window after a switch is never stated. We choose it and report sensitivity to the choice.

**The training target inconsistency.** Documented under *The target* above.

## Replication status

`crdata/player_panel.py` already produces all three player-type features across 11,007,296 players.
Among the 768,410 with 20 or more battles:

| quantity | published | ours, Season 18 |
|---|---|---|
| players | 34,619 | 768,410 |
| switching events | 2,554 in test | 5,627,038 |
| post-loss switch rate | 0.487 / 0.148 by type | 0.239 overall |
| post-win switch rate | 0.064 / 0.066 by type | 0.145 overall |
| loss reactivity | 0.423 / 0.082 by type | +0.094 overall, positive for 57.1% |

**The behavioural claim replicates.** Players switch more after losing, confirmed at 22 times the player count.

**The performance claim does not.** Grouping our players by switch rate gives a win rate spread of 1.53 points against the published 3 to 5, and the relationship is not monotone: win rate rises from never-switching to occasional switching before falling.

Our overall post-win switch rate of 0.145 is more than double either published switching type, which needs explaining before the comparison is trusted.

## The layer we add

Adaptation Cost is the construct the whole method rests on, and no published work measures it.
It cannot be measured without card levels, because a deck switch moves three things at once and only their sum is observable.

**Main question. When a player switches decks, what is the cost actually made of?**

1. **Level cost.** How much of the post-switch dip is moving onto less-upgraded cards? `RESULTS_CARD_LEVEL.md` supplies the conversion.
2. **Strength cost.** How much is the new deck simply weaker? Measured in the outcome-blind space, per D13.
3. **Mastery cost.** What survives 1 and 2, and does `elixir_leaked` confirm it as real degradation? Present on 3,687 of 3,687 battles in the collector test run.
4. **Who pays it.** Does the cost differ across player types and across rank?
5. **Does the headline survive?** Never-switchers hold 0.746 more levels per card than constant-switchers, most of the spread of the entire level curve.
6. **Is `ΔWR` the right target?** It mean-reverts by construction, and matchmaking pins win rate near 50 percent.

Items 1 to 3 decompose the cost.
Items 4 to 6 correct the method.
