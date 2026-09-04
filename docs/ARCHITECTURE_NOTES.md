# Architecture notes

`docs/ARCHITECTURE_NOTES.md` records candidate model designs and questions that must be settled before implementation.

**Status: mostly proposed.**
`docs/DECISIONS.md` D19 settles the first ten-battle next-switch data contract.
`docs/DECISIONS.md` D20 settles the deterministic card vocabulary.
The later encoder, head, loss, and hyperparameter choices in this file remain proposals.

## System shape

The recommendation problem contains three different data structures.

- Recent battles are an ordered sequence with irregular time gaps.
- A deck is an unordered set of eight cards.
- Player summaries, card levels, behavioral subtype, and candidate features are static context.

One shared player-context representation can support several heads, but one homogeneous encoder should not be forced to represent every structure.

```text
Recent battles ------> history encoder ------> player context h
                            |
Old deck -----------> set encoder ------------|
                                               |--> candidate representation
Candidate deck -----> set encoder ------------|
                                               |
Player summary and subtype -------------------|
Card levels and experience -------------------|
                                               |
                             +-----------------+------------------+
                             |                                    |
                    mastery-cost head                     adoption head
                             |                                    |
                      mastery effect                       adoption chance

Candidate and likely opponent decks --> antisymmetric matchup model
                                               |
                                        deck advantage
                                               |
                    deck advantage + mastery effect
                                               |
                                        candidate value
                                               |
                                  rank candidates or Stay
```

The candidate value remains conceptually:

$$
\widehat V(d' \mid x)
=
\widehat{\Delta}_{\text{deck}}(d' \mid x)
+
\widehat{\Delta}_{\text{mastery}}(d' \mid x)
$$

The deployable model must estimate both terms from information available before the recommendation.
The retrospective equations in `docs/EQUATIONS.md` define what the project wants to measure, not yet how an online model produces the estimate.

## First project: recent battle sequence

### Output

The history encoder maps the battles available before decision time to one player-context vector $h_t$.
Every downstream head may read $h_t$, but no downstream decision is allowed to change the history used to construct it.

### Candidate battle token

One token represents one completed battle.
The first next-switch slice uses the fields settled in D19.
Later ablations may remove or replace a field for the final encoder.

- The result is win or loss.
- Crown difference records outcome severity.
- The deck enters through a permutation-invariant deck embedding.
- A deck-change flag records whether the deck differs from the preceding battle.
- Deck-change magnitude records Jaccard distance from the preceding deck.
- The time gap uses `log1p(minutes_since_previous_battle)` or learned time buckets.
- Average deck elixir is computed from the card reference table.
- Total card level and distance from maximum describe investment at that battle.
- Trophies describe the player's trophy-ladder context, not pure skill.
- Game mode is included only if the training population contains more than one allowed mode.

Card identities should not be concatenated in slot order.
The same eight cards must produce the same deck representation under every card permutation.

### Static or long-term context

The short sequence does not have to relearn stable player behavior on every example.
A separate summary branch can provide switch rate, loss reactivity, switch drift, prior deck experience, trophy band, and other leakage-safe aggregates computed strictly before time $t$.

The full-season behavioral subtype cannot be used as an online feature without acknowledging that it contains future behavior.
A deployment-faithful experiment needs a rolling or training-period-only subtype.

### Sequence length

The paper uses the ten most recent matches, but $K=10$ is not settled for this project.
The comparison should include several history lengths and a variable-length mask.
The useful question is not whether a longer window raises training accuracy, but whether it improves held-out player performance without adding future leakage or mostly encoding player identity.

### Time handling

Battles are event sequences rather than evenly sampled measurements.
Ten battles played in one hour should not be treated as equivalent to ten battles spread across a week.

The first baseline should give the time gap to the model as an ordinary token feature.
More specialized continuous-time models are justified only if that baseline leaves a clear, repeatable error pattern related to timing.

## Candidate history encoders

### Non-neural baseline

Use CatBoost or another tree model on explicitly lagged battle features and rolling summaries.
This baseline tests whether learned sequential state contributes beyond readable summaries.
It is not a cheap substitute for the full model and must use the same train, validation, test, and target definitions.

### GRU baseline

A GRU updates one hidden state after each battle and returns the final state as player context.
It is compact, naturally supports variable-length histories, and is the closest replication of the paper.

A bidirectional GRU is valid when it only processes a completed past window because every event in that window is already known at decision time.
A causal single-direction GRU is easier to cache and update online.
Both should be compared if online state reuse matters.

### Temporal convolutional network

A temporal convolutional network applies causal, dilated one-dimensional convolutions over the battle tokens.
It trains in parallel and makes the temporal receptive field explicit.
For short histories, it is a serious alternative to recurrence rather than a secondary novelty model.

Padding and causal masking must prevent the model from seeing events after the decision point.
Kernel size and dilation determine which battle patterns the network can represent.

### Self-attention encoder

A small causal self-attention encoder can learn which earlier battles matter for the current state.
Relative time information should accompany position because the observations are irregularly spaced.

The paper's small GRU advantage over its Transformer does not establish that Transformers fail generally.
The paper only compared two particular implementations on windows of ten matches, where long-range attention has little room to help.

### Advanced continuous-time models

Neural controlled differential equations and neural point-process models represent irregular time directly.
They are substantially more complex and should not be the first implementation for histories this short.
They become relevant if the project predicts behavior continuously between battles or if timing gaps remain a major source of residual error.

## Head notes

### Deck-value computation

The existing permutation-invariant deck encoder and antisymmetric matchup terms should supply the structural deck advantage.
A generic unconstrained MLP should not replace structural antisymmetry.

The candidate score must integrate over an opponent distribution available before recommendation time.
Using the opponents observed after a historical switch is valid for retrospective decomposition but leaks future information into a deployable evaluation.

### Mastery-cost head

The mastery-cost head estimates how the player's execution changes after moving from the old deck to candidate $d'$.
A candidate-conditioned MLP can read player context, old and new deck embeddings, transition distance, prior card experience, and behavioral summaries.

The head should predict uncertainty or quantiles in addition to a mean because post-switch outcomes are noisy and many player-candidate pairs are sparse.
The retrospective intercept in `docs/EQUATIONS.md` becomes infinite for an all-win or all-loss window unless it is regularized or partially pooled.

### Adoption head

The adoption head predicts whether the player is behaviorally likely to use a candidate deck.
CatBoost provides a strong readable baseline, while a candidate-conditioned neural ranker provides an end-to-end alternative.

Adoptability is not the same as value.
A familiar but harmful deck should not outrank an unfamiliar beneficial deck solely because it is easier to adopt.
Adoptability can filter infeasible candidates, impose a minimum threshold, or break near-ties.

### WHO, WHEN, and WHAT

The three decisions can be derived from candidate values instead of trained as three independent hard classifiers.

- WHAT is the candidate with the highest predicted value.
- WHEN is whether the best candidate has reliably positive value now.
- WHO is expressed through player context, predicted mastery cost, and any explicit coverage rule.

One possible conservative rule is:

$$
\widehat V(d^*) - \lambda\widehat\sigma_V(d^*) > 0
$$

The uncertainty multiplier $\lambda$ is a policy choice and is not settled.

### Auxiliary heads

Auxiliary tasks may help the history encoder learn useful state before the scarce transition-quality target is introduced.
Candidate tasks include next-result prediction, next-deck-change prediction, transition-magnitude prediction, crown-difference regression, and rolling behavioral-type prediction.

Auxiliary losses can also cause negative transfer.
Each head needs an ablation showing whether it improves the primary target rather than only its own metric.

### Training and inference

The heads may share the history and deck representations.
Hard PersonaGate or TimingGate decisions should not remove training examples from later heads because early gates would starve them of data and compound errors.
Hard gates belong in policy evaluation or inference unless an experiment demonstrates a benefit from staged training.

Player-level splitting is required so the encoder is tested on players it did not memorize.
Every rolling feature, subtype, baseline, and normalization statistic must be fit without reading beyond the prediction time or outside the training split.

## Proposed experiment order

1. Freeze the prediction time, outcome horizon, target, and player-level data split.
2. Build one canonical sequence-example generator with explicit leakage checks.
3. Fit the lagged-feature tree baseline.
4. Fit the GRU replication baseline.
5. Fit a temporal convolutional network under the same parameter and tuning budget.
6. Fit a small time-aware self-attention model only after the first three results are trustworthy.
7. Inspect calibration, subgroup errors, sensitivity to history length, and errors as a function of time gap.
8. Select the history encoder before beginning the mastery, adoption, or policy heads.

## Reading path

Read these in order through item 6.
Items 7 through 9 are optional extensions once the basic alternatives are clear.

### 1. Cho et al. 2014, Learning Phrase Representations using RNN Encoder-Decoder

[Paper](https://arxiv.org/abs/1406.1078)

This paper introduces the gated recurrent unit and explains how a sequence is compressed into a fixed-length representation.
Think about each battle as an input symbol with several features, while remembering that the project needs only the encoder and not the language decoder.

### 2. Chung et al. 2014, Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling

[Paper](https://arxiv.org/abs/1412.3555)

This paper compares GRUs and LSTMs and finds that both gated units improve over a basic recurrent unit, with neither universally dominant.
Think about the gates as learned choices about which parts of earlier battles to remember or forget, not as proof that recurrence is automatically the right model.

### 3. Hidasi et al. 2015, Session-based Recommendations with Recurrent Neural Networks

[Paper](https://arxiv.org/abs/1511.06939)

This is the closest conceptual analogy because it uses short user-action histories to produce recommendation scores over candidates.
Think about the difference between predicting what a user will choose and predicting which choice will improve the user's outcome, because observed Clash Royale switches are not optimal labels.

### 4. Bai, Kolter, and Koltun 2018, An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling

[Paper](https://arxiv.org/abs/1803.01271)

This paper presents the temporal convolutional network as a general sequence model and compares it against recurrent baselines across many tasks.
Think about receptive field, causal padding, and fair tuning budgets rather than treating the reported aggregate victory as a guarantee for ten-battle histories.

### 5. Kang and McAuley 2018, Self-Attentive Sequential Recommendation

[Paper](https://arxiv.org/abs/1808.09781)

SASRec applies causal self-attention to user histories and balances short-term item transitions with longer-range behavior.
Think about whether attention identifies genuinely useful earlier battles or merely gives a larger model another way to memorize common players and decks.

### 6. Li, Wang, and McAuley 2020, Time Interval Aware Self-Attention for Sequential Recommendation

[Paper](https://doi.org/10.1145/3336191.3371786)

TiSASRec extends sequential self-attention with the time intervals between user actions.
Think about the difference between battle position and elapsed time, especially when ten battles can span one session or several days.

### 7. Choi et al. 2016, RETAIN

[Paper](https://proceedings.neurips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html)

RETAIN uses reverse-time attention over irregular longitudinal records to expose influential visits and variables.
Think about whether a recommendation should identify which past losses or switches affected player context, while remembering that attention weights are model explanations rather than causal effects.

### 8. Seedat et al. 2022, Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations

[Paper](https://proceedings.mlr.press/v162/seedat22b.html)

TE-CDE combines irregular-time sequence modeling with counterfactual outcome estimation.
Think about the similarity between asking what would happen under a treatment and asking what would happen under a deck switch, but do not import its causal claims without checking the project's treatment-assignment assumptions.

### 9. Lim et al. 2021, Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting

[Paper](https://arxiv.org/abs/1912.09363)

The Temporal Fusion Transformer combines static covariates, recurrent local processing, attention, feature selection, and quantile forecasts.
Think about its separation of static player context from time-varying battle inputs and its uncertainty outputs, while recognizing that the full architecture is probably oversized for a ten-event sequence.
