# Baseline ladder results, Season 18 ladder battles

Data: `BattlesStaging_01042021_WL_tagged.csv`, 1,105,943 ladder battles from 4 January 2021.
Split: time-ordered, 75 percent train and 25 percent test.
Orientation randomised before fitting, per assumption A10 in `ASSUMPTIONS.md`.

## Results

| model | features | accuracy | AUC |
|---|---|---|---|
| coin flip | 0 | 50.06 | 0.5000 |
| trophy difference | 1 | 51.30 | 0.5199 |
| card identity, logistic regression | 102 | 51.80 | 0.5312 |
| card identity, gradient boosting | 204 | 55.05 | 0.5714 |
| card level difference, logistic regression | 1 | 55.80 | 0.5925 |
| card identity plus card level, gradient boosting | 103 | 58.25 | 0.6199 |

Rows using gradient boosting were fitted on a 400,000-battle subsample for tractability.
The card level rung scores 55.46 on the full file and 55.80 on the subsample, so subsampling moves results by about 0.3 points.

## Reading the results

Card identity carries real signal.
A gradient boosted model on card identity alone reaches 55.05 percent against a 50 percent baseline.

Card identity needs interactions to show that signal.
Logistic regression on card identity reaches only 51.80 percent, so a linear model understates deck composition by roughly 3 points.
Any comparison that uses a linear card model alone will unfairly diminish deck composition.

One feature beats the whole card identity model.
Total card level difference, a single number, reaches 55.80 percent against 55.05 percent for a 204-feature gradient boosted card identity model.

Card identity and account investment are separately sufficient.
Either information source alone produces accuracy in the 55 percent range.
Combining both reaches 58.25 percent, which exceeds the 56.94 and 57.25 percent reported by the work under critique.

## What the results support

The claim that deck composition predicts outcomes is true but incomplete.
An account investment proxy predicts outcomes at least as well as deck composition on this data.
A study that reports only a deck model, without testing an investment baseline, cannot distinguish the two.

## What the results do not support

The results do not show that deck composition is worthless.
Card identity beats the baseline by about 5 points once interactions are modelled.

The results do not transfer directly to the work under critique.
That work used War Day battles from a later era with a different card pool, so the comparison is illustrative rather than a replication.

Card level and skill are entangled, with a correlation of +0.786 between card level and trophies.
Attributing the card level effect entirely to spending overstates the case.
