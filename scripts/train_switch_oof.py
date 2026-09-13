"""Generate player-level out-of-fold probabilities from the switch model."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crdata.switch_dataset import NO_PROPENSITY_FOLD, load_switch_arrays
from models.switch_model import SwitchPredictionModel
from scripts.train_switch_baseline import (
    HEAD_HIDDEN_DIM,
    atomic_json_dump,
    atomic_torch_save,
    cpu_state_dict,
    environment_metadata,
    evaluate,
    metrics,
    optimizer_to_device,
    prepare_arrays,
    resolve_device,
    restore_rng_state,
    seed_everything,
    select_batch,
    tensors_on_device,
)


def atomic_numpy_save(values: np.ndarray, path: Path) -> None:
    """Write one NumPy array atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values)
    temporary.replace(path)


def fold_signature(
    arguments: argparse.Namespace, fold: int, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Return settings that must match for a fold run to resume."""
    return {
        "fold": fold,
        "seed": arguments.seed,
        "max_epochs": arguments.max_epochs,
        "patience": arguments.patience,
        "min_delta": arguments.min_delta,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "head_hidden_dim": HEAD_HIDDEN_DIM,
        "summary_feature_set": "current_deck_count",
        "metadata_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def train_fold(
    fold: int,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    arguments: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    """Train without one player fold and predict every row in that fold."""
    signature = fold_signature(arguments, fold, metadata)
    stem = f"fold_{fold}"
    best_path = arguments.output_dir / f"{stem}.pt"
    progress_path = arguments.output_dir / f"{stem}_progress.pt"
    result_path = arguments.output_dir / f"{stem}_result.json"
    logits_path = arguments.output_dir / f"{stem}_heldout_logits.npy"

    if result_path.exists() and logits_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("training") != signature:
            raise ValueError(
                f"completed fold {fold} has different settings; "
                "choose another output directory"
            )
        print(f"fold {fold}: already complete", flush=True)
        return result, np.load(logits_path)

    fold_ids = np.asarray(arrays["propensity_folds"])
    splits = np.asarray(arrays["splits"])
    training_indices = np.flatnonzero((splits == 0) & (fold_ids != fold))
    heldout_indices = np.flatnonzero((splits == 0) & (fold_ids == fold))
    validation_indices = np.flatnonzero(splits == 1)
    prepared, preprocessing = prepare_arrays(arrays, training_indices)
    tensors = tensors_on_device(prepared, device)

    seed_everything(arguments.seed)
    model = SwitchPredictionModel(
        embedding_rows=int(metadata["embedding_rows"]),
        level_mean=float(preprocessing["level_mean"]),
        level_standard_deviation=float(preprocessing["level_standard_deviation"]),
        head_hidden_dim=HEAD_HIDDEN_DIM,
        summary_dim=int(tensors["summary_features"].shape[-1]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1

    if progress_path.exists():
        checkpoint = torch.load(progress_path, map_location="cpu", weights_only=False)
        if checkpoint.get("training") != signature:
            raise ValueError(
                f"partial fold {fold} has different settings; "
                "choose another output directory"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer_to_device(optimizer, device)
        history = checkpoint["history"]
        best_loss = float(checkpoint["best_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint)
        print(f"fold {fold}: resuming at epoch {start_epoch}", flush=True)

    print(
        f"fold {fold}: {len(training_indices):,} train, "
        f"{len(heldout_indices):,} held out",
        flush=True,
    )
    stopped_early = epochs_without_improvement >= arguments.patience
    for epoch in range(start_epoch, arguments.max_epochs + 1):
        if stopped_early:
            break
        model.train()
        order = training_indices.copy()
        np.random.default_rng(arguments.seed + fold * 100_000 + epoch).shuffle(order)
        loss_sum = 0.0
        started = time.perf_counter()
        for start in range(0, len(order), arguments.batch_size):
            batch = select_batch(
                tensors, order[start:start + arguments.batch_size], device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*batch[:-1]), batch[-1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch[-1])

        validation, _ = evaluate(
            model, tensors, validation_indices, arguments.batch_size, device
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / len(order),
            "validation_log_loss": validation["log_loss"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        improved = validation["log_loss"] < best_loss - arguments.min_delta
        if improved:
            best_loss = validation["log_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "model_state": cpu_state_dict(model),
                    "fold": fold,
                    "epoch": epoch,
                    "validation_log_loss": best_loss,
                    "data": metadata,
                    "preprocessing": preprocessing,
                    "training": signature,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "training": signature,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": cuda_rng_states,
            },
            progress_path,
        )
        print(
            f"  epoch {epoch:02d}: train={record['train_loss']:.4f} "
            f"validation={record['validation_log_loss']:.4f} "
            f"best={best_loss:.4f} time={record['seconds']:.1f}s",
            flush=True,
        )
        if epochs_without_improvement >= arguments.patience:
            stopped_early = True
            print("  early stopping", flush=True)
            break

    if not best_path.exists():
        raise RuntimeError(f"fold {fold} did not produce a best checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    validation, _ = evaluate(
        model, tensors, validation_indices, arguments.batch_size, device
    )
    heldout, heldout_logits = evaluate(
        model, tensors, heldout_indices, arguments.batch_size, device
    )
    result = {
        "fold": fold,
        "training": signature,
        "train_examples": len(training_indices),
        "heldout_examples": len(heldout_indices),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "validation": validation,
        "heldout": heldout,
        "history": history,
        "checkpoint": best_path.name,
        "logits": logits_path.name,
    }
    atomic_numpy_save(heldout_logits, logits_path)
    atomic_json_dump(result, result_path)
    return result, heldout_logits


def probability_summary(probabilities: np.ndarray) -> dict[str, Any]:
    """Summarize raw propensity coverage without choosing a policy threshold."""
    quantiles = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    return {
        "quantiles": {
            str(quantile): float(np.quantile(probabilities, quantile))
            for quantile in quantiles
        },
        "fraction_between_0.05_and_0.95": float(
            np.mean((probabilities >= 0.05) & (probabilities <= 0.95))
        ),
        "fraction_between_0.10_and_0.90": float(
            np.mean((probabilities >= 0.10) & (probabilities <= 0.90))
        ),
        "fraction_between_0.20_and_0.80": float(
            np.mean((probabilities >= 0.20) & (probabilities <= 0.80))
        ),
        "fraction_between_0.40_and_0.60": float(
            np.mean((probabilities >= 0.40) & (probabilities <= 0.60))
        ),
    }


def validate_player_fold_membership(
    player_indices: np.ndarray, splits: np.ndarray, fold_ids: np.ndarray
) -> None:
    """Reject a cache that places one training player in multiple folds."""
    fold_by_player: dict[int, int] = {}
    for player, split, fold in zip(player_indices, splits, fold_ids):
        if split != 0:
            continue
        player_id = int(player)
        fold_id = int(fold)
        previous = fold_by_player.setdefault(player_id, fold_id)
        if previous != fold_id:
            raise ValueError(f"training player {player_id} appears in multiple folds")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create five player-level out-of-fold switch probabilities."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    if arguments.max_epochs < 1 or arguments.patience < 1 or arguments.batch_size < 1:
        parser.error("max epochs, patience, and batch size must be positive")
    if arguments.min_delta < 0.0:
        parser.error("min delta cannot be negative")
    if arguments.learning_rate <= 0.0 or arguments.weight_decay < 0.0:
        parser.error("learning rate must be positive and weight decay cannot be negative")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    device = resolve_device(arguments.device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    arrays, metadata = load_switch_arrays(arguments.cache)
    required = {"next_wins", "propensity_folds"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"cache is missing required arrays: {sorted(missing)}")
    if metadata.get("continuity") != "same_collection":
        raise ValueError("out-of-fold training requires a same_collection cache")
    if metadata.get("summary_feature_set") != "current_deck_count":
        raise ValueError("out-of-fold training requires the selected nine-feature cache")

    fold_metadata = metadata.get("propensity_folds", {})
    fold_count = int(fold_metadata.get("count", 0))
    if fold_count != 5:
        raise ValueError("out-of-fold training requires exactly five propensity folds")
    fold_ids = np.asarray(arrays["propensity_folds"])
    splits = np.asarray(arrays["splits"])
    validate_player_fold_membership(
        np.asarray(arrays["player_indices"]), splits, fold_ids
    )
    if not np.all(fold_ids[splits != 0] == NO_PROPENSITY_FOLD):
        raise ValueError("validation and test rows must not belong to propensity folds")
    if set(np.unique(fold_ids[splits == 0])) != set(range(fold_count)):
        raise ValueError("training rows do not cover every declared propensity fold")

    oof_logits = np.full(len(splits), np.nan, dtype=np.float32)
    runs = []
    for fold in range(fold_count):
        result, heldout_logits = train_fold(
            fold, arrays, metadata, arguments, device
        )
        heldout_indices = np.flatnonzero((splits == 0) & (fold_ids == fold))
        if len(heldout_logits) != len(heldout_indices):
            raise RuntimeError(f"fold {fold} returned the wrong number of predictions")
        oof_logits[heldout_indices] = heldout_logits
        runs.append(result)
        atomic_numpy_save(oof_logits, arguments.output_dir / "oof_switch_logits.npy")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    training_indices = np.flatnonzero(splits == 0)
    if not np.isfinite(oof_logits[training_indices]).all():
        raise RuntimeError("some training examples are missing out-of-fold predictions")
    if np.isfinite(oof_logits[splits != 0]).any():
        raise RuntimeError("out-of-fold predictions must be restricted to training rows")
    probabilities = np.full_like(oof_logits, np.nan)
    probabilities[training_indices] = 1.0 / (
        1.0 + np.exp(-np.clip(oof_logits[training_indices], -30.0, 30.0))
    )
    labels = np.asarray(arrays["labels"])[training_indices]
    report = {
        "data": metadata,
        "environment": environment_metadata(device),
        "calibration": "raw_uncalibrated",
        "prediction_scope": "original training examples only",
        "runs": runs,
        "pooled_oof": metrics(labels, oof_logits[training_indices]),
        "probabilities": probability_summary(probabilities[training_indices]),
    }
    atomic_numpy_save(probabilities, arguments.output_dir / "oof_switch_probabilities.npy")
    atomic_json_dump(report, arguments.output_dir / "results.json")
    print("\nPOOLED OUT-OF-FOLD RESULTS", flush=True)
    for name, value in report["pooled_oof"].items():
        print(f"  {name:18s} {value:.6f}", flush=True)
    print(f"written {arguments.output_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
