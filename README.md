# clash2

When you win in Clash Royale, what are you actually winning with?
And of whatever deck effect remains, can we measure how much better some decks are than others?


this project attempts to: (measure aspects like skill) + create a recommendation algorithm that decides when a player should switch decks based off their recent battle history and behavioural type, and which deck to switch to.

## How a battle is predicted

Every battle is written as the log-odds that side A wins.
Four heads add up on the log-odds scale, and each head flips sign on its own when the two players swap sides.

$$
\text{logit } P(A \text{ wins}) = \text{skill} + \text{investment} + \text{deck strength} + \text{counters}
$$

- **skill** is one number per player, in `MatchupModel.player_skill`.
- **investment** is an odd function of the card level gap, in `MatchupModel.investment`.
- **deck strength** is how good a deck is against the field, in `MatchupModel.strength_head`.
- **counters** is how a deck fares against one specific other deck, in `MatchupModel.counter_term`.

## How a deck switch is predicted

Write $\mathrm{new}_i$ and $\mathrm{old}_i$ for the log-odds the model gives the new deck and the old deck against the same opponent in battle $i$, at the card levels the player would really hold.
A player switches decks, then plays $n$ more battles.

The deck part is the average gap between the two.

$$
\Delta_{\text{deck}} = \frac{1}{n} \sum_{i=1}^{n} \big( \mathrm{new}_i - \mathrm{old}_i \big)
$$

The piloting part is one shift that makes predicted wins match real wins, where $y_i$ is 1 for a win and 0 for a loss.

$$
\sum_{i=1}^{n} \sigma\big( \mathrm{new}_i + \delta_{\text{after}} \big) = \sum_{i=1}^{n} y_i
\qquad
\Delta_{\text{mastery}} = \delta_{\text{after}} - \delta_{\text{usual}}
$$

The objective function adds the two parts.

$$
V = \Delta_{\text{deck}} + \Delta_{\text{mastery}}
$$

Recommend the deck with the largest $V$, and only when that largest $V$ is above 0.
Recommend staying otherwise.

`docs/EQUATIONS.md` carries the full definitions and the assumption the objective function depends on.
