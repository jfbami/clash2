# Project conventions

Read `CLAUDE.md` first for the working agreement.
Read `DECISIONS.md` for what has been settled and when.
Read `REJECTED.md` before proposing an approach, so a ruled-out idea is not raised again.

## Terminology

Use these words exactly.
Mixing them silently changes what a result means.

**Side A and side B** name the two players in a battle.
Side A is not the winner.
The Season 18 source files store the winner first, and that layout is destroyed during loading.

**Investment** means account progression: card levels, and in the current game evolutions.
Never call it "pay to win" in code or documentation.
Investment mixes money and time, and the data cannot separate them.

**Skill** means player ability.
Trophies are a proxy for skill, not skill itself.
Matchmaking equalises trophies before a battle, so within-match trophy variance is near zero by design.

**Transitive** deck effect means some decks are simply stronger than others.
**Cyclic** deck effect means the advantage depends on the opposing deck.
The cyclic part is what a Bradley-Terry or Elo model cannot represent.

**Counter term** is the antisymmetric bilinear form in `crdata/neural.py`.
It is the blade-chest model of Chen and Joachims (2016).

**Baseline ladder** is the nested sequence of deliberately limited models in `RESULTS_BASELINE_LADDER.md`.
It is unrelated to the game's trophy ladder.
When ambiguity is possible, write "trophy ladder" for the game mode.

**Outcome-blind** means built without any win or loss information.
The card space in `crdata/embedding.py` is outcome-blind, and must stay that way.

**Archetype** means a region of deck space, not a labelled category.
There is no definitive archetype list.

## Architecture constraints

**Antisymmetry is structural, never learned.**
Swapping side A and side B must negate the predicted log-odds exactly.
`crdata/neural.py` verifies this numerically rather than assuming it.
Any new model term must negate under a side swap or it does not go in.

**The archetype space must never see outcomes.**
Defining archetypes with win and loss data and then measuring archetype strength is circular.
`crdata/embedding.py` uses deck co-occurrence only.

**Orientation must never carry signal.**
Storage order, column position, and player tag order have all leaked the label at some point in this project.
Any new loader states explicitly how orientation is assigned and shows the resulting label rate is 50 percent.

**Everything additive happens on the log-odds scale.**
Components add there and do not add on the probability scale.
Variance decomposition is only meaningful on log-odds.
Convert to win rate for presentation, never for arithmetic.

**Stream, do not load.**
The machine has 16.6 GB of RAM with roughly 1.4 GB free.
The season is 22 GB of CSV and 1.7 GB of Parquet.
Any code touching the full season reads in batches and holds only aggregates.

**Filter at modelling time, not collection time.**
`crdata/parse.py` and `crdata/etl.py` flag rows rather than dropping them.
Dropping rows during ingestion has already caused one silent bias in this project.

## How to explain a change

State what changed, why, and what would break if it is wrong.
Name the exact file and function.

Report the number that would embarrass the result, not only the number that supports it.
If a figure came from a subsample, a single day, or a different setting than the surrounding text, say so next to the figure.

When a result depends on a choice, give the number under both choices.
The variance decomposition differs by a factor of four between the linear and neural models, and reporting only one would be misleading.

Corrections go in the open.
If a previously reported number was wrong, state the old value, the new value, and what caused the error.

## How decisions get made

The user decides the research question, all modelling decisions, and the prose.
Claude lays out options with tradeoffs and waits.

Every settled decision goes in `DECISIONS.md` with a date.
Every ruled-out approach goes in `REJECTED.md` with the reason it was ruled out.

A decision is not settled because Claude made it while building something.
Surface those explicitly so they can be reversed.

## Verification expectations

A structural property is verified numerically, not asserted.
Antisymmetry, permutation invariance, and label balance all have checks.

A prediction about the data is written down before the code runs.
The card space checks in `scripts/build_card_space.py` were stated in advance for this reason.

An assumption that can be cheaply tested is tested, not assumed.
`ASSUMPTIONS.md` records which of the ten are testable.
