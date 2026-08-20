# Identifying assumptions and acknowledged biases

Every estimate in this project rests on assumptions that cannot all be verified.
Each assumption below states what the assumption buys, what breaks if the assumption is false, and whether the assumption can be cheaply tested.

Testability matters here.
This project criticises prior work for leaving an identifying assumption unstated and unchecked.
Assuming away something testable invites the same criticism.

## A1. End-of-season representativeness

Assume the last day of Season 18 reflects ladder play generally.

Assuming A1 permits use of a single convenient 1.1M-battle file.

A1 breaks if end-of-season trophy pushing shifts deck choice toward safe meta decks.
A1 also breaks if the active player mix skews toward committed players.

A1 is testable and cheap.
Pull one mid-season day and compare deck, level, and trophy distributions.

## A2. Card level proxies account investment

Assume summed card level indexes time and money put into an account.

Assuming A2 supports the central claim that a measured deck signal is partly a progression proxy.

A2 breaks if card level primarily reflects something else.
Card level correlates +0.786 with trophies, so card level is jointly an investment measure and a skill measure.

Within-player fixed effects absorb the account-constant portion of card level.
Report card level as a joint measure rather than a clean one.

## A3. Ladder generalises to other modes

Assume deck effects estimated on ladder resemble deck effects in war and tournament play.

A3 breaks if level normalisation changes the mechanism.
Ladder is the most investment-confounded mode, so pay-to-win estimates from ladder are an upper bound.

A3 is not testable with Season 18 data alone.
The collector in `scripts/collect.py` covers other modes.

## A4. Trophies proxy player skill

Assume `startingTrophies` indexes player skill.

A4 breaks if trophies mostly track time played.
Matchmaking also equalises trophies, leaving little within-match variance.
Winners average 4870.3 trophies against losers at 4869.7.

Use trophies as a covariate and rely on within-player fixed effects as the primary skill control.

## A5. Deck switching is exogenous within a player

Assume a player's deck change is not driven by something that also drives the outcome.

Assuming A5 supports the entire within-player identification strategy, which makes A5 the load-bearing assumption.

A5 breaks if players switch decks after losing streaks or switch to chase the meta.
Both behaviours are plausible and would bias deck effects.

A5 is partially testable.
Check whether deck switches correlate with recent results and whether post-switch win rates revert to the mean.

## A6. Matchmaking ignores decks and card levels

Assume pairing is trophy-based only.

A6 breaks if matchmaking considers card level, because observed pairings would then be selected on the treatment.

Evidence weakly supports A6.
Trophies are balanced across sides at 4870.3 against 4869.7, while card levels are not balanced at 95.53 against 94.61.

## A7. Stationarity within the estimation window

Assume no balance patch falls inside the estimation window.

A7 is safe for a single day.
Recheck A7 per window across a season.

## A8. The sample represents more than top active players

Assume findings extend beyond the crawled population.

A8 breaks if deck effects differ at lower trophy levels.

A8 is testable.
The observed trophy range is 24 to 7685, so estimates can be stratified by arena or trophy band.

## A9. Bot and smurf contamination is negligible

Assume contamination is negligible.

A9 is unverified and acknowledged as unknown.

## A10. Orientation randomisation is required

A10 is a processing requirement rather than an assumption.

The Season 18 source files are sorted winner-first, with `winner.crowns` greater than `loser.crowns` in 100.00 percent of rows.
A model trained without randomising side assignment reaches 100 percent accuracy from column position alone.
Randomise orientation before fitting any model.
