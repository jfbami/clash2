# Deck switch objective

`docs/EQUATIONS.md` holds the objective function the deck-switch recommender maximises.
The objective function says whether a player should change decks, and which deck to change to.

**Status: proposed, not settled.**
No entry in `docs/DECISIONS.md` covers the objective function in `docs/EQUATIONS.md`.
No code in `crdata/` computes the objective function.
Read `docs/ASSUMPTIONS.md` before drawing any conclusion from the objective function.

## Words used in this file

**Log-odds** is a way of writing a win chance.
Log-odds of 0 means a coin flip.
Positive log-odds means more likely to win.
Log-odds are used instead of percentages because log-odds add up and percentages do not.
`docs/CONVENTIONS.md` states this rule.

**The old deck** is the deck the player used before the switch.

**The new deck** is the deck the player used after the switch.

**Sigma**, written as $\sigma$, turns log-odds back into a win chance between 0 and 1.

## How to compute the objective function

A player switches from the old deck to the new deck, then plays $n$ more battles.
Number those battles $1$ to $n$.

1. For each battle $i$, ask the model the win log-odds for the **new** deck against the opponent in battle $i$.
   Call the answer $\mathrm{new}_i$.
2. For the same battle $i$, ask the model the win log-odds for the **old** deck against that same opponent.
   Call the answer $\mathrm{old}_i$.
3. Use the card levels the player would really hold for each deck.
   A player who switches to an archetype they never upgraded gets low card levels in step 1, and the model already knows what low card levels cost.
4. Average the gap between the two answers to get the deck part of the objective function.
5. Fit one number that shifts all the predictions until they match what actually happened, to get the piloting part of the objective function.
6. Add the deck part and the piloting part together.

## The deck part

Average the gap between the two answers from steps 1 and 2.

$$
\Delta_{\text{deck}} = \frac{1}{n} \sum_{i=1}^{n} \big( \mathrm{new}_i - \mathrm{old}_i \big)
$$

Both answers use the same opponent.
So $\Delta_{\text{deck}}$ does not depend on who the player happened to face.

## The piloting part

Let $y_i$ be what actually happened in battle $i$.
$y_i$ is 1 for a win and 0 for a loss.

Find the single number $\delta_{\text{after}}$ that makes the predicted number of wins equal the real number of wins.

$$
\sum_{i=1}^{n} \sigma\big( \mathrm{new}_i + \delta_{\text{after}} \big) \;=\; \sum_{i=1}^{n} y_i
$$

A negative $\delta_{\text{after}}$ means the player did worse than the decks alone predicted.
Solving the equation above gives the same answer as fitting $\delta_{\text{after}}$ by log loss, so no accuracy is lost by using the simpler form.

Now compute $\delta_{\text{usual}}$ the same way, using the player's other battles instead.

$$
\Delta_{\text{mastery}} = \delta_{\text{after}} - \delta_{\text{usual}}
$$

Leave out the battles just before the switch when computing $\delta_{\text{usual}}$.
A player switches after a bad run, and a bad run is usually followed by a normal run even when nothing has changed.
Including the bad run would put that false recovery back into $\Delta_{\text{mastery}}$.

## The objective function

$$
V = \Delta_{\text{deck}} + \Delta_{\text{mastery}}
$$

Compute $V$ once for every deck the player could switch to.
Recommend the deck with the largest $V$, but only when that largest $V$ is above 0.
Recommend staying when no deck scores above 0.

Staying always scores exactly 0.
Staying means the new deck and the old deck are the same deck, so every gap in $\Delta_{\text{deck}}$ is $\mathrm{old}_i - \mathrm{old}_i = 0$.
Nothing has to be estimated to get that 0, so nothing can bias it.

## The assumption the objective function depends on

The model must score a matchup that never happened.
Step 2 asks for the old deck against opponents the player only met after switching away from the old deck.
Every value of $\Delta_{\text{deck}}$ rests on that guess.
