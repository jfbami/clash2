# clash2

When you win in Clash Royale, what are you actually winning with?
can we measure how much better some decks are than others?


this project attempts to: (measure aspects like skill) + create a recommendation algorithm that decides when a player should switch decks based off their recent battle history and behavioural type, and which deck to switch to.

## Colab switch baseline

Copy or sync this folder to `MyDrive/clash2`, including the ignored
`data/switch_continuity_same_collection/arrays` cache. Then open
`scripts/train_switch_baseline_colab.ipynb` from Google Drive with Colab and run
all cells using a GPU runtime. Training progress and results are saved back to
Drive, so an interrupted run can resume.

After establishing that baseline, use
`scripts/train_current_deck_count_colab.ipynb` for the controlled ablation that
adds total observed usage of the player's current exact deck.

Once that feature is selected, `scripts/train_switch_oof_colab.ipynb` generates
five player-level out-of-fold switch-propensity predictions for later outcome
modelling.

