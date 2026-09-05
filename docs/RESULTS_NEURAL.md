# Neural matchup model results

Model: `crdata/models/matchup.py`, trained by `scripts/train_matchup.py`.
Data: `BattlesStaging_01042021_WL_tagged.csv`, 1,105,943 Season 18 ladder battles.
Split: time-ordered, 75 percent train and 25 percent test.

## Architecture

Four antisymmetric terms summed on the log-odds scale.

```
logit P(A wins) = skill + investment + deck strength + counters
```

`DeckEncoder` sum-pools card embeddings then applies an MLP, so the model can represent synergy.
An additive per-card strength cannot express "these two cards are worth more together than apart".

`MatchupModel.counter_term` scores a pair of decks with the blade-chest antisymmetric form.
A single strength number per deck forces a total order and cannot represent a cycle.

Structural properties verified numerically rather than assumed:

- Antisymmetry: `max |f(A,B) + f(B,A)|` is exactly 0.0.
- Self-counter: `max |counter(D,D)|` is exactly 0.0.
- Permutation invariance: 1.5e-4, which is float32 rounding.

## Performance

| model | accuracy | AUC | log loss |
|---|---|---|---|
| original study, reported | 56.94 | 0.5968 | not reported |
| logistic regression, cards plus level | 51.80 | 0.5312 | 0.6909 |
| gradient boosting, cards plus level | 58.25 | 0.6199 | not recorded |
| neural, with player skill embedding | 59.70 | 0.6397 | 0.6595 |
| neural, skill embedding removed | 60.53 | 0.6496 | 0.6550 |

## Ablation

Each row retrained from scratch on 500,000 battles, best epoch by log loss.

| variant | accuracy | AUC | log loss |
|---|---|---|---|
| full model | 59.24 | 0.6327 | 0.6623 |
| no counter term, Bradley-Terry only | 56.86 | 0.6050 | 0.6713 |
| no player skill embedding | 59.31 | 0.6351 | 0.6614 |
| no investment term | 55.56 | 0.5836 | 0.6777 |
| counter rank 2 | 58.78 | 0.6304 | 0.6633 |
| counter rank 24 | 59.39 | 0.6359 | 0.6617 |

Removing the counter term costs 2.38 accuracy points.
A model without counters scores 56.86 percent, which is within noise of the 56.94 percent the original study reported.

Removing the player skill embedding **improves** the model.
With about 2.3 battles per player in a single day, a per-player parameter is fitted on noise.
The skill term is therefore dropped from the final model.

## Player identity leakage check

The test set was split by whether each player was seen during training.

| group | battles | accuracy | AUC |
|---|---|---|---|
| both players seen in training | 45,410 | 56.69 | 0.5967 |
| exactly one seen | 99,140 | 58.86 | 0.6268 |
| neither seen | 131,936 | 61.56 | 0.6645 |

Accuracy is **higher** on unseen players, which is the opposite of identity leakage.
The likely cause is population composition: frequent players appear in training more often and are more evenly matched, so their battles are harder to predict.

## Variance decomposition

Player skill term removed, so the shares below cover only the three remaining components.

| component | share |
|---|---|
| counters, cyclic | 37.9 |
| deck strength, transitive | 32.5 |
| account investment | 29.7 |

Deck effect in total is 70.3 percent.
Within that deck effect, 46.2 percent is transitive and 53.8 percent is cyclic.

**More than half of the deck effect is matchup rather than raw strength.**
A Bradley-Terry or Elo style model discards that half by construction.

## How much better are the best decks

| comparison | win rate for the stronger side |
|---|---|
| 75th against 25th percentile deck | 54.90 |
| 95th against 5th percentile deck | 61.58 |
| 90th percentile counter matchup | 65.21 |
| 99th percentile counter matchup | 75.32 |

A strong counter matchup moves outcomes far more than raw deck strength does.

## Caveats

The decomposition is highly sensitive to model capacity.
A linear model attributes 79.7 percent to investment and 18.5 percent to deck.
The neural model attributes 29.7 percent to investment and 70.3 percent to deck.
The deck encoder holds an MLP while the investment term is a constrained odd function on one scalar, so some of that shift is capacity asymmetry rather than a discovery.
Report both numbers rather than the flattering one.

All results rest on a single end-of-season day, which is assumption A1 in `ASSUMPTIONS.md` and is still untested.

The skill share cannot be estimated from this data at all, because matchmaking equalises trophies and one day gives too few battles per player.
Estimating skill requires the longitudinal collector.
