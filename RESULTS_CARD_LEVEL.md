# How much do card levels matter?

![Win rate against card level advantage](figures/card_level_effect.png)

Data: 36,865,856 Season 18 ladder battles, 7 December 2020 to 3 January 2021.
Built by `scripts/plot_card_level_effect.py`, counting logic in `crdata/level_effect.py`.
4 January 2021 is excluded, per D9 in `DECISIONS.md`.

## Reading the picture

The horizontal axis is how many levels higher your cards are than your opponent's, averaged over the 8 cards in each deck.
Every card in this data sits on a 1 to 13 scale where 13 means fully upgraded.
The average card in Season 18 sits at 11.63, and 51.00 percent of all cards played are already maxed.

The vertical axis is how often you win.
The grey bars show how often each gap actually comes up.

| your average card level advantage | your win rate | share of battles |
|---|---|---|
| 0.00 | 49.97% | 24.84% |
| +0.25 | 53.64% | 5.48% |
| +0.50 | 57.33% | 3.52% |
| +1.00 | 65.06% | 1.78% |
| +1.50 | 72.03% | 0.76% |
| +2.00 | 77.39% | 0.28% |
| +2.50 | 79.05% | 0.10% |

## The headline

One card level takes you from a coin flip to **65 percent**.

Half a level is already worth 57 percent.
Most players never see a gap that large: 85.82 percent of battles happen within one level either way, and a quarter of all battles are dead even.

## The effect is steady, then it stalls

From +0.25 to +1.50, each extra level buys almost exactly the same amount.
Measured on the log-odds scale, the effect runs between 0.58 and 0.63 per average card level with no trend.

Past +1.50 it flattens.
An advantage of +2.50 wins 79.05 percent, barely better than the 77.39 percent at +2.00.
Once you are that far ahead the opponent is already losing, and more levels cannot push much further.

This lines up with the model that was fitted independently.
A logistic regression fitted a weight of 0.068384 per summed level, which is 0.547 per average card level.
Counting gives 0.58 to 0.63 across the range where most battles happen.
The fitted number is slightly low because a straight line has to average through the flat tail.

## This is a floor, not a ceiling

Matchmaking pairs players on trophies, and it pairs them tightly: 99.86 percent of battles are between players within 50 trophies.
So both players in every one of these battles are at the same rung of the ladder.

That matters for how you read the curve.
If two players sit at the same trophy count and one has a full card level less, that player had to be better at the game to get there.
The 65 percent therefore already has a skill handicap working against it.

The true value of a card level advantage is **higher** than this curve shows, not lower.

## What this does not show

**It does not separate card levels from deck choice.**
Rarities have different starting points: a Common can sit at level 1, but a Legendary never drops below 9.
So part of any level gap is which cards you chose to play rather than how much you invested.
Untangling those two is the next piece of work.

**It does not say levels are the only thing that matters.**
Three quarters of battles are within 0.625 levels, and something decides those.
This curve says nothing about what.

**It measures power, not money.**
A level is a level here.
Dragging a Legendary from 9 to 13 costs vastly more than dragging a Common from 1 to 13, so equal advantages on this axis represent very unequal amounts of spending.
See A2 in `ASSUMPTIONS.md`.

**It is Season 18.**
December 2020, before evolutions existed.
Nothing here transfers to the current game without checking.

## Why this is counted rather than modelled

Every point on the curve is an observed win rate, not a prediction.
Group the battles by advantage, count the wins, divide.

`RESULTS_NEURAL.md` records why that matters.
Logistic regression gives card level 79.7 percent of the explained variance and the neural model gives it 29.7 percent, on the same data.
That four-fold gap is a property of the models, not of Clash Royale.
A counted win rate has no model behind it, so it has nothing to disagree about.

## Two choices behind the figure

**The axis is an average, not a sum.**
The underlying count bins on the summed level difference, which is the integer the data actually holds.
Dividing by 8 is a constant and changes no shape, but +1.00 reads as "my cards are one level better than yours" where +8 needs a footnote.

**One line was cut.**
`crdata.level_effect.trophy_matched_win_rate` reweights each trophy bracket to the season-wide mix, and an earlier draft plotted it alongside the counted curve.
It was cut because it is misleading at the edges.
At +2.50 it reports 84.43 percent against the counted 79.05 percent, and that gap comes from brackets holding a few hundred battles, where the answer is driven by who ends up there rather than by card levels.
The two agree to within 0.1 points across the range covering 85 percent of battles.
The function stays in `crdata/level_effect.py` and is printed by `scripts/plot_card_level_effect.py` as a check.
