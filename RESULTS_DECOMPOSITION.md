# What a Clash Royale win is made of

Data: `BattlesStaging_01042021_WL_tagged.csv`, 400,000 ladder battles from 4 January 2021.
Model: `crdata/decompose.py`, a separable logit fitted by `fit_additive_model`.

    logit P(side A wins) = skill + investment + deck

Skill enters as the trophy difference, investment as the total card level difference, and deck as summed per-card strength.

## Variance shares

| component | share of explained variance |
|---|---|
| account investment, card levels | 79.7 |
| deck design, card identity | 18.5 |
| player skill, via trophies | 1.8 |

Fitted weights: trophy +0.003461 per trophy, card level +0.068384 per level.

## How much better are the best decks

| comparison | win rate for the stronger deck |
|---|---|
| 75th percentile deck against 25th | 54.27 |
| 95th percentile deck against 5th | 60.56 |
| strongest observed deck against weakest | 76.29 |

The 25th to 75th percentile spread is 0.1714 log-odds.

## Strongest and weakest cards, net of card level

Strongest: Bowler, Cannon Cart, Lava Hound, Skeleton Barrel, Night Witch, The Log, Skeleton Army, Barbarian Barrel.

Weakest: Elixir Golem, Mirror, Archers, Minion Horde, Clone, Wizard, Giant, Magic Archer.

## Three caveats that change how the shares read

The skill share of 1.8 percent does not mean skill barely matters.
Matchmaking equalises trophies before a battle starts, so almost no skill gap survives into the data.
Winners average 4870.3 trophies against losers at 4869.7.
The decomposition measures what varies between two already matched opponents, not what determines outcomes in general.

The deck share of 18.5 percent is a lower bound.
`fit_additive_model` is additive in cards, so the model cannot represent synergy or counters.
Gradient boosting recovers about 3 points more accuracy from card identity than a linear model, so a model with interactions would raise the deck share.

The investment share is not purely spending.
Card level correlates +0.786 with trophies, so the level term carries skill as well as investment.
See assumption A2 in `ASSUMPTIONS.md`.
