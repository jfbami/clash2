"""Train the neural matchup model on a Season 18 ladder file.

Usage:  python scripts/train_matchup.py [--subsample N] [--epochs N]

Reports test accuracy, AUC and log loss, then decomposes the predicted log-odds
into skill, investment, deck strength and counters.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from crdata.models.matchup import MatchupModel
from crdata.season18 import as_index_arrays, load_randomised

SEASON18_CSV = Path(
    r"C:\Users\jfbaa\AppData\Local\Temp\claude"
    r"\C--Users-jfbaa-OneDrive-Documents-clash2"
    r"\d24c6794-c5fc-463a-925a-588dd12c92e6\scratchpad\kaggle_sample"
    r"\BattlesStaging_01042021_WL_tagged.csv")


def to_tensors(data, rows: slice) -> dict[str, torch.Tensor]:
    return {
        "cards_a": torch.from_numpy(data.cards_a[rows]),
        "cards_b": torch.from_numpy(data.cards_b[rows]),
        "player_a": torch.from_numpy(data.player_a[rows]),
        "player_b": torch.from_numpy(data.player_b[rows]),
        "level_a": torch.from_numpy(data.level_a[rows]),
        "level_b": torch.from_numpy(data.level_b[rows]),
    }


def evaluate(model: MatchupModel, batch: dict, labels: np.ndarray) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(**batch)).numpy()
    return (float(((probabilities > 0.5) == labels).mean()),
            float(roc_auc_score(labels, probabilities)),
            float(log_loss(labels, probabilities)))


def report_decomposition(model: MatchupModel, batch: dict) -> None:
    model.eval()
    with torch.no_grad():
        parts = model.components(**batch)

    variances = {
        "skill": float(parts.skill.var()),
        "investment": float(parts.investment.var()),
        "deck strength": float(parts.deck.var()),
        "counters": float(parts.counters.var()),
    }
    total = sum(variances.values())

    print("\n" + "=" * 62)
    print("VARIANCE DECOMPOSITION of predicted log-odds")
    print("=" * 62)
    for name, value in sorted(variances.items(), key=lambda item: -item[1]):
        print(f"  {name:16s} {value:8.4f}   {100 * value / total:5.1f} percent")

    transitive = variances["deck strength"]
    cyclic = variances["counters"]
    deck_total = transitive + cyclic
    print(f"\n  Within the deck effect:")
    print(f"    transitive, some decks are better : {100 * transitive / deck_total:5.1f} percent")
    print(f"    cyclic, it depends who you face   : {100 * cyclic / deck_total:5.1f} percent")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=400_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--counter-rank", type=int, default=8)
    arguments = parser.parse_args()

    print("loading ...")
    battles = load_randomised(SEASON18_CSV, subsample=arguments.subsample)
    data = as_index_arrays(battles)
    n_battles = len(data)
    cut = int(n_battles * 0.75)
    print(f"  {n_battles:,} battles | {len(data.card_ids)} cards | {data.n_players:,} players")
    print(f"  train {cut:,} / test {n_battles - cut:,}, time ordered\n")

    train_batch = to_tensors(data, slice(0, cut))
    test_batch = to_tensors(data, slice(cut, n_battles))
    train_labels = torch.from_numpy(data.side_a_won[:cut]).float()
    test_labels = data.side_a_won[cut:]

    torch.manual_seed(0)
    model = MatchupModel(
        n_cards=len(data.card_ids), n_players=data.n_players,
        counter_rank=arguments.counter_rank)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    skill_parameters = list(model.player_skill.parameters())
    other_parameters = [p for p in model.parameters()
                        if all(p is not s for s in skill_parameters)]
    optimiser = torch.optim.AdamW([
        {"params": other_parameters, "weight_decay": 1e-5},
        {"params": skill_parameters, "weight_decay": 1e-2},
    ], lr=3e-3)
    criterion = nn.BCEWithLogitsLoss()

    order = np.arange(cut)
    best_logloss, best_state = float("inf"), None

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        np.random.default_rng(epoch).shuffle(order)
        started = time.time()
        for start in range(0, cut, arguments.batch_size):
            rows = torch.from_numpy(order[start:start + arguments.batch_size])
            optimiser.zero_grad()
            batch = {key: value[rows] for key, value in train_batch.items()}
            loss = criterion(model(**batch), train_labels[rows])
            loss.backward()
            optimiser.step()

        accuracy, auc, logloss = evaluate(model, test_batch, test_labels)
        marker = ""
        if logloss < best_logloss:
            best_logloss = logloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"  epoch {epoch:2d}  acc={accuracy * 100:5.2f}%  auc={auc:.4f}  "
              f"logloss={logloss:.4f}  {time.time() - started:4.0f}s{marker}")

    model.load_state_dict(best_state)
    accuracy, auc, logloss = evaluate(model, test_batch, test_labels)
    print(f"\nBEST MODEL  acc={accuracy * 100:.2f}%  auc={auc:.4f}  logloss={logloss:.4f}")
    report_decomposition(model, test_batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
