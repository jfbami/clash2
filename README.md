# clash2

When you win in Clash Royale, what are you actually winning with?
can we measure how much better some decks are than others?


this project attempts to: (measure aspects like skill) + create a recommendation algorithm that decides when a player should switch decks based off their recent battle history and behavioural type, and which deck to switch to.

## Architecture

`B` means batch size. The smaller line in each box shows the tensor dimension.

### Switch-propensity model — built

The implemented model uses battles 1–10 to estimate whether the player will
change their exact deck in battle 11.

```mermaid
%%{init: {"theme":"base","flowchart":{"htmlLabels":true,"curve":"basis"},"themeVariables":{"fontFamily":"monospace","lineColor":"#3f3f46","primaryTextColor":"#27272a"}}}%%
flowchart TB
    history(["Battles 1–10<br/><small>B × 10 chronological battles</small>"])

    cards["Card IDs<br/><small>B × 10 × 8</small>"]
    levels["Displayed card levels<br/><small>B × 10 × 8</small>"]
    battleRaw["Six recorded battle features<br/><small>B × 10 × 6</small>"]
    summaryRaw["Nine long-term player features<br/><small>B × 9</small>"]

    embed["Card embedding<br/><small>B × 10 × 8 × 24</small>"]
    levelStd["Level standardization<br/><small>B × 10 × 8 × 1</small>"]
    cardJoin["Concatenate card information<br/><small>B × 10 × 8 × 25</small>"]
    cardMLP["Shared card MLP + GELU<br/><small>25 → 48 → 48 per card</small>"]
    sumPool["Sum pool over 8 cards<br/><small>B × 10 × 48</small>"]
    deckMLP["Deck MLP + GELU<br/><small>48 → 96 → 48 per battle</small>"]

    battlePrep["Standardize + trophies-missing flag<br/><small>6 recorded → 7 model features</small>"]
    battleJoin["Concatenate deck + battle context<br/><small>48 + 7 = 55 per battle</small>"]
    gru["One-layer forward GRU<br/><small>10 × 55 → final B × 64</small>"]

    summaryPrep["Summary standardization<br/><small>B × 9</small>"]
    contextJoin["Concatenate recent + long-term context<br/><small>64 + 9 = B × 73</small>"]
    switchHead["Switch MLP + GELU + dropout<br/><small>73 → 128 → 1 logit</small>"]
    sigmoid["Sigmoid<br/><small>B × 1 probability</small>"]
    prediction(["P(switch deck in battle 11)<br/><small>one probability per sequence</small>"])

    target["Observed battle-11 action<br/><small>B binary labels: stay 0 / switch 1</small>"]
    loss["Binary cross-entropy with logits<br/><small>one training loss</small>"]

    history --> cards
    history --> levels
    history --> battleRaw
    history --> summaryRaw

    cards --> embed
    levels --> levelStd
    embed --> cardJoin
    levelStd --> cardJoin
    cardJoin --> cardMLP --> sumPool --> deckMLP

    battleRaw --> battlePrep
    deckMLP --> battleJoin
    battlePrep --> battleJoin
    battleJoin --> gru

    summaryRaw --> summaryPrep
    gru --> contextJoin
    summaryPrep --> contextJoin
    contextJoin --> switchHead
    switchHead --> sigmoid --> prediction

    switchHead -. training .-> loss
    target -. training .-> loss

    classDef source fill:#eef8fc,stroke:#83b7cc,color:#27272a,stroke-width:1px;
    classDef operation fill:#f2efff,stroke:#8b6cf6,color:#27272a,stroke-width:1px;
    classDef sequence fill:#fff7dc,stroke:#d5a72e,color:#27272a,stroke-width:1px;
    classDef output fill:#ebf8ef,stroke:#56a86c,color:#27272a,stroke-width:1px;
    classDef training fill:#fff1e8,stroke:#d6814b,color:#27272a,stroke-width:1px;

    class history,cards,levels,battleRaw,summaryRaw source;
    class embed,levelStd,cardJoin,cardMLP,sumPool,deckMLP,battlePrep,battleJoin,summaryPrep,contextJoin,switchHead operation;
    class gru sequence;
    class sigmoid,prediction output;
    class target,loss training;
```

### Player-level out-of-fold propensity training — built

Five copies of the model produce leakage-safe switch probabilities for the
future outcome phase. Every player's sequences remain together in one fold.

```mermaid
%%{init: {"theme":"base","flowchart":{"htmlLabels":true,"curve":"basis"},"themeVariables":{"fontFamily":"monospace","lineColor":"#3f3f46","primaryTextColor":"#27272a"}}}%%
flowchart TB
    players(["Original training pool<br/><small>4,027 players · 110,419 sequences</small>"])
    folds["Player-level five-fold split<br/><small>all sequences from one player stay together</small>"]

    m0["Fold model 0<br/><small>train folds 1–4 · predict fold 0</small>"]
    m1["Fold model 1<br/><small>train folds 0,2,3,4 · predict fold 1</small>"]
    m2["Fold model 2<br/><small>train folds 0,1,3,4 · predict fold 2</small>"]
    m3["Fold model 3<br/><small>train folds 0,1,2,4 · predict fold 3</small>"]
    m4["Fold model 4<br/><small>train folds 0–3 · predict fold 4</small>"]

    merge["Align held-out predictions<br/><small>B = 110,419 raw probabilities</small>"]
    oof(["Out-of-fold switch propensity<br/><small>model never trained on predicted player</small>"])

    players --> folds
    folds --> m0
    folds --> m1
    folds --> m2
    folds --> m3
    folds --> m4
    m0 --> merge
    m1 --> merge
    m2 --> merge
    m3 --> merge
    m4 --> merge
    merge --> oof

    classDef source fill:#eef8fc,stroke:#83b7cc,color:#27272a,stroke-width:1px;
    classDef operation fill:#f2efff,stroke:#8b6cf6,color:#27272a,stroke-width:1px;
    classDef output fill:#ebf8ef,stroke:#56a86c,color:#27272a,stroke-width:1px;

    class players source;
    class folds,m0,m1,m2,m3,m4,merge operation;
    class oof output;
```

### Outcome phase — proposed, not built

```mermaid
%%{init: {"theme":"base","flowchart":{"htmlLabels":true,"curve":"basis"},"themeVariables":{"fontFamily":"monospace","lineColor":"#3f3f46","primaryTextColor":"#27272a"}}}%%
flowchart TB
    x(["Information before battle 11<br/><small>battles 1–10 + player context</small>"])
    propensity["Built switch-propensity model<br/><small>e(X) = one switch probability</small>"]
    encoder["Proposed win-outcome encoder<br/><small>representation dimension not selected</small>"]
    stay["Proposed stay head<br/><small>mu0(X) = one win probability</small>"]
    switchOutcome["Proposed switch head<br/><small>mu1(X) = one win probability</small>"]
    delta["Compare outcome probabilities<br/><small>delta(X) = mu1(X) − mu0(X)</small>"]
    policy(["Future recommendation<br/><small>stay · consider switching · insufficient evidence</small>"])

    x --> propensity
    x --> encoder
    encoder --> stay
    encoder --> switchOutcome
    stay --> delta
    switchOutcome --> delta
    delta --> policy
    propensity --> policy

    classDef built fill:#ebf8ef,stroke:#56a86c,color:#27272a,stroke-width:1px;
    classDef planned fill:#f2efff,stroke:#8b6cf6,color:#27272a,stroke-width:1px,stroke-dasharray:5 4;
    classDef output fill:#fff7dc,stroke:#d5a72e,color:#27272a,stroke-width:1px,stroke-dasharray:5 4;

    class x,propensity built;
    class encoder,stay,switchOutcome,delta planned;
    class policy output;
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the feature definitions,
implementation map, and additional notes.
