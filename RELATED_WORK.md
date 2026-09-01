# Related work

What each paper bears on in this project, and how deeply it has been read.

Read depth is recorded because a paper skimmed from an abstract cannot be cited for a specific number.
`full` means the text was read.
`abstract` means only the abstract or a summary was read.

Add a paper here when it changes a decision, not when it is merely adjacent.

---

## The framing under critique

### Janusz, Grad and Grzegorowski 2019, Clash Royale Challenge: How to Select Training Decks for Win-rate Prediction

FedCSIS 2019, DOI 10.15439/2019F365.
Read depth: full.

A data mining competition summary.
Participants selected ten training subsets of 600 to 1500 decks, sized in steps of 100, to train radial-kernel support vector regression models predicting deck win-rate.
Data was 100,000 decks from three consecutive league seasons of 1v1 ladder, derived from over 160,000,000 game results via RoyaleAPI.
Each row held eight card names, games played, number of players using the deck, and an estimated win-rate.
Scoring was R-squared averaged across the ten subsets, evaluated on decks popular in the three seasons after the training period.
The winning team scored 0.2552 against a baseline of 0.1564, from 115 teams across 18 countries.

**Bears on the critique, not the method.**
The dataset carries no card level and no player identity, so account investment and player skill cannot be separated in it at all.
That is the same disqualifying gap recorded for the s1m0n38 dataset in `REJECTED.md` R1.

**The unit is the deck, not the battle.**
A deck win-rate is marginal across every opponent it faced and every player who used it.
Player skill and account investment are therefore baked into the target variable rather than controlled for.
Marginalising over opponents also removes the matchup structure that `MatchupModel.counter_term` in `crdata/neural.py` exists to measure.

**Cite 0.2552 only with its caveat.**
That figure is R-squared on aggregated deck win-rate.
It is not comparable to the battle-outcome accuracies in `RESULTS_BASELINE_LADDER.md`, which score a different target.

**The authors report their own target as unstable across seasons.**
They note the same deck appears in training and test data with different win-rates, because balance changes and players adapt.
That is assumption A7 in `ASSUMPTIONS.md` appearing in published work.

**The authors also criticise their own winners.**
They write that no top-10 team used an approach applicable in practice, because the winning search heuristics tuned against validation data that would not exist in deployment.
The paper is candid, which suits the generous-critique posture this project has chosen.

---

## Attribution and its limits

### Hypothesis Class Determines Explanation: Why Accurate Models Disagree on Feature Attribution

arXiv 2603.15821.
Read depth: abstract.

Prediction-equivalent models produce different feature attributions.
Across 24 datasets and 93,510 pairwise comparisons, 35.4 percent of prediction-equivalent model pairs disagree substantially on feature rankings, at Spearman correlation below 0.5.
The mechanism is representational: a linear model SHAP value collapses to a weight times a centred feature and cannot encode interactions, while tree and neural classes carry interaction terms.

**Explains the four-fold swing recorded in `RESULTS_NEURAL.md`.**
A linear model attributed 79.7 percent of explained variance to investment.
The neural model in `crdata/neural.py` attributes 29.7 percent on the same data.
That gap is the documented phenomenon rather than a defect in either model.

The paper proposes an Explanation Reliability Score, averaging pairwise attribution agreement across an ensemble of prediction-equivalent models, treating scores below 0.5 as unreliable.

### The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and Complete Under Collinearity

arXiv 2605.21492.
Read depth: abstract.

No feature ranking can be simultaneously faithful, stable, and complete when features are collinear.
The attribution ratio diverges as 1 over 1 minus rho squared.

**Applies directly to separating skill from investment.**
Assumption A2 in `ASSUMPTIONS.md` records a correlation of +0.786 between card level and trophies.
That places the inflation factor at 2.6 for the skill and investment pair specifically.
The honest output for that pair may be a tie rather than a split.

The proposed remedy, DASH, reports ties for symmetric features on the grounds that collinear features cannot be distinguished.

### Lundberg and Lee 2017, A Unified Approach to Interpreting Model Predictions

NeurIPS 2017.
Read depth: abstract.

SHAP, the additive feature attribution framework both papers above critique.
Relevant as the machine learning framing of a complete ablation over the skill, investment and deck information sets.

---

## Architecture

### Zaheer et al. 2017, Deep Sets

NeurIPS 2017.
Read depth: abstract.

Sum-pooling followed by an MLP is the universal form for permutation-invariant set functions.
That is exactly `DeckEncoder` in `crdata/neural.py`, so the architecture choice has a proof behind it rather than an intuition.

The universal approximation result requires a latent dimension of order N to the power D for multisets of D-dimensional vectors.
That bound bears on whether `embed_dim=48` supplies enough capacity for 8-card decks.

### Chen and Joachims 2016, Modeling Intransitivity in Matchup and Comparison Data

WSDM 2016.
Read depth: abstract.

The blade-chest model, implemented as `MatchupModel.counter_term` in `crdata/neural.py`.
Represents each item with multiple vectors so intransitive matchups can be expressed, which a single strength scalar cannot.

---

## Transitive and cyclic structure

### Balduzzi et al. 2019, Open-ended Learning in Symmetric Zero-sum Games

ICML 2019, arXiv 1901.08106.
Read depth: abstract plus Theorem 1.

Theorem 1 decomposes any antisymmetric payoff function orthogonally into a transitive gradient component and a cyclic divergence-free component.
For finite populations the Schur decomposition factors the antisymmetric matrix as W J W-transpose, with 2x2 blocks whose singular values measure cyclic strength.

**Offers a unique split where this project currently has a model-dependent one.**
The 46.2 against 53.8 transitive-cyclic split in `RESULTS_NEURAL.md` is read off the variances of two learned heads, which are not orthogonal by construction.

### Czarnecki et al. 2020, Real World Games Look Like Spinning Tops

NeurIPS 2020, arXiv 2004.09468.
Read depth: abstract.

Real games have a spinning-top geometry: an upright axis of transitive strength, and a radial axis of cyclic dimension that is widest at intermediate strength and narrow at both extremes.

Testable here by stratifying on trophy band, which assumption A8 in `ASSUMPTIONS.md` already contemplates.

---

## Prior decompositions in other games

### Player Skill Decomposition in Multiplayer Online Battle Arenas

arXiv 1702.06253.
Read depth: abstract.

Decomposes League of Legends and DOTA2 match outcomes into character base attributes, player base ability, and champion-specific expertise.
DOTA2 outcomes are dominated by hero selection while League of Legends depends on both hero and player.

**Names a component this project does not model.**
Champion-specific expertise is player-loadout fit, which is neither pure skill nor pure deck.
`MatchupModel` in `crdata/neural.py` has no equivalent player-deck interaction term.

### Identifying and Clustering Counter Relationships of Team Compositions in PvP Games for Efficient Balance Analysis

arXiv 2408.17180.
Read depth: abstract.

Bradley-Terry for strength ratings plus vector quantization for counter relationships, validated on Age of Empires II, Hearthstone, Brawl Stars and League of Legends.
A learned codebook is an alternative to clustering the outcome-blind card space built by `crdata/embedding.py`.

---

## Player behaviour

### When to Switch, Not Just What: Transition Quality Prediction in Clash Royale

arXiv 2605.21868.
Read depth: abstract.

926,334 matches from 34,619 Clash Royale players.
Reports that deck switches are triggered by losses and that frequent switchers have lower win rates.

**Evidence against assumption A5 in `ASSUMPTIONS.md`.**
A5 assumes deck switching is exogenous within a player, which is what any within-player identification of deck effects rests on.
