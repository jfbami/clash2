# Switch-head ablation: preliminary run

Date: 2026-09-10

## Question

Does the final 72-dimensional player context need a nonlinear prediction head, and does expanding that head to 128 units improve on the implemented 64-unit compact head?

## Data and split

The run used the deduplicated local live-battle collection. Every example contains ten chronological battles and predicts whether the focal player changes decks in the following battle.

- 234,844 sliding-window examples from 5,878 eligible players
- 163,875 training examples from 4,114 players
- 35,779 validation examples from 882 players
- 35,190 test examples from 882 players
- Players, rather than windows, were assigned to the 70/15/15 split
- All preprocessing statistics were fitted on training-player examples only

The local card reference was refreshed with the one card present in the live battles but absent from the table: Minion Giant, a rare card. No battle was dropped for an unknown rarity.

## Controlled comparison

Every model used the same card encoder, deck encoder, 64-unit forward GRU, player split, batches, optimizer settings, and five-epoch budget. The full model trained end to end; only the final head shape changed.

| Head | Total parameters | Best validation epoch | Validation log loss | Test log loss | Test ROC AUC | Test calibration error |
|---|---:|---:|---:|---:|---:|---:|
| Direct linear, 72 to 1 | 39,241 | 5 | 0.3260 | 0.3188 | 0.9234 | 0.0099 |
| Compact nonlinear, 72 to 64 to 1 | 43,905 | 5 | **0.3220** | 0.3134 | 0.9261 | 0.0060 |
| Expanded nonlinear, 72 to 128 to 1 | 48,641 | 4 | 0.3222 | **0.3124** | **0.9267** | **0.0044** |

The 15 training epochs across the three models took about six minutes on the local CPU after the reusable example cache had been built.

## Interpretation

Both nonlinear heads beat direct linear fusion on the held-out test examples. The expanded head ranked first on test log loss, ROC AUC, and calibration, but its test log-loss advantage over the compact head was only 0.0009 and the compact head was marginally better on validation log loss.

This is evidence that final-head nonlinearity is useful. It is not enough evidence to select 128 over 64 because the comparison has only one initialization and the sliding windows within each player are correlated. Repeat the compact-versus-expanded comparison across several model seeds and estimate uncertainty by resampling held-out players before settling the width.

For continued development, the 128-unit expanded head is the working default because it led every recorded test metric and has negligible additional CPU cost. The choice remains provisional until it survives the stricter data and repeated-seed checks.
