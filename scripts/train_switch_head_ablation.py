"""Train linear, compact, and expanded heads on identical player splits."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crdata.battle_features import fit_battle_feature_standardization
from crdata.card_levels import fit_level_standardization
from crdata.models.switch_model import SwitchPredictionModel
from crdata.summary_features import fit_summary_feature_standardization
from crdata.switch_dataset import build_switch_array_cache, load_switch_arrays


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEADS = (("linear", None), ("compact_64", 64), ("expanded_128", 128))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Return ten-bin absolute calibration error weighted by bin size."""
    edges = np.linspace(0.0, 1.0, 11)
    total = len(labels)
    error = 0.0
    for index in range(10):
        lower, upper = edges[index], edges[index + 1]
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        if selected.any():
            error += selected.mean() * abs(
                float(probabilities[selected].mean()) - float(labels[selected].mean())
            )
    return float(error)


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return {
        "log_loss": float(log_loss(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "calibration_error": expected_calibration_error(labels, probabilities),
    }


def make_batch(arrays: dict[str, np.ndarray], indices: np.ndarray) -> tuple[torch.Tensor, ...]:
    """Copy one disk-backed batch into CPU tensors with model dtypes."""
    return (
        torch.from_numpy(np.asarray(arrays["cards"][indices])).long(),
        torch.from_numpy(np.asarray(arrays["levels"][indices])).float(),
        torch.from_numpy(np.asarray(arrays["battle_features"][indices])).float(),
        torch.from_numpy(np.asarray(arrays["summary_features"][indices])).float(),
        torch.from_numpy(np.asarray(arrays["labels"][indices])).float(),
    )


def transformed_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fit preprocessing on training players and return disk arrays plus transforms."""
    training = np.flatnonzero(arrays["splits"] == 0)
    level_stats = fit_level_standardization(arrays["levels"][training])
    battle_stats = fit_battle_feature_standardization(arrays["battle_features"][training])
    summary_stats = fit_summary_feature_standardization(arrays["summary_features"][training])
    return {
        **arrays,
        "battle_features": battle_stats.transform(arrays["battle_features"]),
        "summary_features": summary_stats.transform(arrays["summary_features"]),
        "level_mean": level_stats.mean,
        "level_standard_deviation": level_stats.standard_deviation,
    }


def evaluate(
    model: SwitchPredictionModel,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = make_batch(arrays, indices[start:start + batch_size])
            outputs.append(model(*batch[:-1]).numpy())
    logits = np.concatenate(outputs)
    labels = np.asarray(arrays["labels"][indices])
    return metrics(labels, logits), logits


def train_one(
    name: str,
    head_hidden_dim: int | None,
    arrays: dict[str, np.ndarray],
    embedding_rows: int,
    epochs: int,
    batch_size: int,
    seed: int,
    output_dir: Path,
) -> dict:
    """Train one head shape and restore its best validation checkpoint."""
    torch.manual_seed(seed)
    model = SwitchPredictionModel(
        embedding_rows=embedding_rows,
        level_mean=float(arrays["level_mean"]),
        level_standard_deviation=float(arrays["level_standard_deviation"]),
        head_hidden_dim=head_hidden_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_indices = np.flatnonzero(arrays["splits"] == 0)
    validation_indices = np.flatnonzero(arrays["splits"] == 1)
    test_indices = np.flatnonzero(arrays["splits"] == 2)
    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = train_indices.copy()
        np.random.default_rng(seed + epoch).shuffle(order)
        loss_sum = 0.0
        started = time.perf_counter()
        for start in range(0, len(order), batch_size):
            batch = make_batch(arrays, order[start:start + batch_size])
            optimizer.zero_grad()
            loss = criterion(model(*batch[:-1]), batch[-1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch[-1])

        validation, _ = evaluate(model, arrays, validation_indices, batch_size)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / len(order),
            "validation_log_loss": validation["log_loss"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"  {name:12s} epoch {epoch}: train={record['train_loss']:.4f} "
            f"validation={record['validation_log_loss']:.4f} "
            f"time={record['seconds']:.1f}s",
            flush=True,
        )
        if validation["log_loss"] < best_loss:
            best_loss = validation["log_loss"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    validation, _ = evaluate(model, arrays, validation_indices, batch_size)
    test, _ = evaluate(model, arrays, test_indices, batch_size)
    torch.save(best_state, output_dir / f"{name}.pt")
    return {
        "name": name,
        "head_hidden_dim": head_hidden_dim,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(np.argmin([row["validation_log_loss"] for row in history]) + 1),
        "validation": validation,
        "test": test,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "battles")
    parser.add_argument(
        "--card-reference", type=Path,
        default=PROJECT_ROOT / "data" / "reference" / "cards.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "switch_head_ablation",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--rebuild-cache", action="store_true")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error("epochs and batch size must be positive")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    cache = arguments.output_dir / "arrays"
    if arguments.rebuild_cache or not (cache / "metadata.json").exists():
        print("building deduplicated sliding-window cache ...", flush=True)
        metadata = build_switch_array_cache(
            arguments.data_root, arguments.card_reference, cache, arguments.seed
        )
    else:
        _, metadata = load_switch_arrays(cache)
    print(json.dumps(metadata, indent=2), flush=True)

    arrays, metadata = load_switch_arrays(cache)
    arrays = transformed_arrays(arrays)
    results = []
    for name, hidden_dim in HEADS:
        results.append(train_one(
            name=name,
            head_hidden_dim=hidden_dim,
            arrays=arrays,
            embedding_rows=int(metadata["embedding_rows"]),
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
            output_dir=arguments.output_dir,
        ))

    report = {"data": metadata, "epochs": arguments.epochs, "heads": results}
    result_path = arguments.output_dir / "results.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nTEST RESULTS")
    for result in results:
        test = result["test"]
        print(
            f"  {result['name']:12s} logloss={test['log_loss']:.4f} "
            f"auc={test['roc_auc']:.4f} calibration={test['calibration_error']:.4f}"
        )
    print(f"written {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
