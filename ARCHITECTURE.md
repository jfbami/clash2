# Model architecture

This document separates the model that is already implemented from the outcome
and recommendation components that are still being designed.

## Current switch-propensity model — built

The current model predicts whether the player will use a different exact deck in
the next eligible battle. It does not predict whether switching is beneficial.

The following image is traced directly from the implemented PyTorch forward pass
using [VisualTorch](https://github.com/willyfh/visualtorch):

![VisualTorch graph of the implemented switch model](architecture_assets/switch-model-visualtorch.png)

The simplified data-flow view below names the domain concepts represented by
those layers:

```mermaid
flowchart LR
    cards["Card IDs<br/>B × 10 battles × 8 cards"]
    levels["Displayed card levels<br/>B × 10 × 8"]
    battle["Battle features<br/>B × 10 × 7"]
    summary["Player summary features<br/>B × 9"]

    embed["Learned card embedding<br/>24 values per card"]
    levelstd["Standardized level<br/>1 value per card"]
    cardmlp["Shared card MLP<br/>25 → 48 → 48"]
    sumpool["Sum over 8 cards<br/>order-invariant deck vector"]
    deckmlp["Deck MLP<br/>48 → 96 → 48"]
    token["Concatenate each deck with<br/>7 battle features<br/>55-value battle token"]
    gru["Forward GRU over 10 battles<br/>55 → 64 final history vector"]
    fuse["Concatenate history and summary<br/>64 + 9 = 73 values"]
    head["Switch head<br/>73 → 128 → 1 logit<br/>GELU + dropout"]
    probability["Sigmoid<br/>P(switch deck in battle 11)"]

    cards --> embed
    levels --> levelstd
    embed --> cardmlp
    levelstd --> cardmlp
    cardmlp --> sumpool --> deckmlp --> token
    battle --> token
    token --> gru --> fuse
    summary --> fuse
    fuse --> head --> probability
```

The seven per-battle model features are result, crown difference, deck-change
indicator, switch magnitude, time gap, trophies, and a trophies-missing flag.
The nine summary features describe longer-term switching behavior and current
deck familiarity.

Sum pooling intentionally makes the eight card positions interchangeable. Card
identity and level still affect the deck vector; only arbitrary card ordering is
discarded.

## Leakage-safe out-of-fold training — built

```mermaid
flowchart TB
    players["All original training players"]
    split["Assign every player and all of their<br/>sequences to exactly one of 5 folds"]
    f0["Model 0<br/>train folds 1–4<br/>predict fold 0"]
    f1["Model 1<br/>train folds 0, 2, 3, 4<br/>predict fold 1"]
    f2["Model 2<br/>train folds 0, 1, 3, 4<br/>predict fold 2"]
    f3["Model 3<br/>train folds 0, 1, 2, 4<br/>predict fold 3"]
    f4["Model 4<br/>train folds 0–3<br/>predict fold 4"]
    oof["Combined raw OOF switch probabilities<br/>one prediction per training example<br/>from a model that never trained on that player"]

    players --> split
    split --> f0
    split --> f1
    split --> f2
    split --> f3
    split --> f4
    f0 --> oof
    f1 --> oof
    f2 --> oof
    f3 --> oof
    f4 --> oof
```

## Next outcome phase — proposed, not built

The next phase would use battle-11 win/loss as the observed outcome. The diagram
is conceptual: the exact outcome-head architecture and propensity adjustment
have not been selected yet.

```mermaid
flowchart LR
    x["Pre-battle information X<br/>battles 1–10 + player context"]
    action["Observed action A<br/>stay = 0 or switch = 1"]
    win["Observed outcome Y<br/>battle-11 win or loss"]

    propensity["Existing frozen switch model<br/>e(X) = P(switch | X)"]
    outcomeencoder["New win-outcome encoder<br/>architecture under discussion"]
    stay["Stay outcome head<br/>mu0(X) = P(win if staying)"]
    switch["Switch outcome head<br/>mu1(X) = P(win if switching)"]
    factual["Training loss uses only the<br/>head for the observed action"]
    compare["Estimated difference<br/>delta(X) = mu1(X) - mu0(X)"]
    decision["Future policy<br/>stay / consider switching / insufficient evidence"]

    x --> propensity
    x --> outcomeencoder
    outcomeencoder --> stay
    outcomeencoder --> switch
    action --> factual
    win --> factual
    stay --> factual
    switch --> factual
    stay --> compare
    switch --> compare
    propensity --> decision
    compare --> decision
```

Only the observed action and outcome are available during training. For example,
if a player stayed and won, the stay head receives that win label; there is no
invented label for what would have happened after switching.

Because no replacement deck is specified, the proposed switch outcome means the
average result of switching in the way similar players historically switched. It
does not represent the effect of switching to a particular deck.

## Implementation map

- Card and deck encoding: `models/card_encoder.py`
- Battle-token assembly: `models/battle_encoder.py`
- GRU and switch head: `models/history_encoder.py`
- End-to-end switch model: `models/switch_model.py`
- Sequential example construction: `crdata/sequences.py`
- Player splits and propensity folds: `crdata/switch_dataset.py`
- OOF switch training: `scripts/train_switch_oof.py`
