"""Train the action-conditioned battle-11 win model from random initialization."""
from __future__ import annotations

import argparse
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

from crdata.switch_dataset import load_switch_arrays
from models.outcome_model import DEFAULT_OUTCOME_HEAD_HIDDEN_DIM, OutcomePredictionModel
from scripts.train_switch_baseline import (
    atomic_json_dump,
    atomic_torch_save,
    cpu_state_dict,
    environment_metadata,
    metrics,
    optimizer_to_device,
    prepare_arrays,
    resolve_device,
    restore_rng_state,
    seed_everything,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 17


def outcome_tensors_on_device(
    arrays: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    """Materialize inputs, observed actions, and observed outcomes on one device."""
    values = {
        "cards": np.asarray(arrays["cards"]).astype(np.int64),
        "levels": np.asarray(arrays["levels"]).astype(np.float32),
        "battle_features": np.asarray(arrays["battle_features"], dtype=np.float32),
        "summary_features": np.asarray(arrays["summary_features"], dtype=np.float32),
        "actions": np.asarray(arrays["labels"]).astype(np.float32),
        "outcomes": np.asarray(arrays["next_wins"]).astype(np.float32),
    }
    return {
        name: torch.from_numpy(array).to(device) for name, array in values.items()
    }


def select_outcome_batch(
    tensors: dict[str, torch.Tensor], indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Select one factual outcome batch from device-resident tensors."""
    index = torch.as_tensor(indices, dtype=torch.long, device=device)
    return (
        tensors["cards"].index_select(0, index),
        tensors["levels"].index_select(0, index),
        tensors["battle_features"].index_select(0, index),
        tensors["summary_features"].index_select(0, index),
        tensors["actions"].index_select(0, index),
        tensors["outcomes"].index_select(0, index),
    )


def factual_metrics(
    outcomes: np.ndarray, logits: np.ndarray, actions: np.ndarray
) -> dict[str, Any]:
    """Report factual performance overall and for each observed action."""
    report: dict[str, Any] = {
        "overall": {
            "examples": int(len(outcomes)),
            "win_rate": float(np.mean(outcomes)),
            **metrics(outcomes, logits),
        }
    }
    for action, name in ((0, "stay"), (1, "switch")):
        selected = actions == action
        if not selected.any():
            raise ValueError(f"evaluation split has no observed {name} examples")
        report[name] = {
            "examples": int(np.sum(selected)),
            "win_rate": float(np.mean(outcomes[selected])),
            **metrics(outcomes[selected], logits[selected]),
        }
    return report


def evaluate(
    model: OutcomePredictionModel,
    tensors: dict[str, torch.Tensor],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate only outcomes observed under each example's factual action."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = select_outcome_batch(
                tensors, indices[start:start + batch_size], device
            )
            outputs.append(model(*batch[:-1]).detach().cpu())
    logits = torch.cat(outputs).numpy()
    index = torch.as_tensor(indices, dtype=torch.long, device=device)
    outcomes = tensors["outcomes"].index_select(0, index).detach().cpu().numpy()
    actions = tensors["actions"].index_select(0, index).detach().cpu().numpy()
    return factual_metrics(outcomes, logits, actions), logits


def validate_player_splits(player_indices: np.ndarray, splits: np.ndarray) -> None:
    """Reject caches that place any player in more than one data split."""
    split_by_player: dict[int, int] = {}
    for player, split in zip(player_indices, splits):
        player_id = int(player)
        split_id = int(split)
        previous = split_by_player.setdefault(player_id, split_id)
        if previous != split_id:
            raise ValueError(f"player {player_id} appears in multiple data splits")


def training_signature(
    arguments: argparse.Namespace, seed: int, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Return settings that must match before a run may resume."""
    return {
        "seed": seed,
        "max_epochs": arguments.max_epochs,
        "patience": arguments.patience,
        "min_delta": arguments.min_delta,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "head_hidden_dim": DEFAULT_OUTCOME_HEAD_HIDDEN_DIM,
        "initialization": "random",
        "loss": "factual_binary_cross_entropy",
        "summary_feature_set": "current_deck_count",
        "metadata_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def train_seed(
    seed: int,
    arrays: dict[str, np.ndarray],
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    preprocessing: dict[str, Any],
    arguments: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Train one random initialization and restore its best validation model."""
    signature = training_signature(arguments, seed, metadata)
    stem = f"outcome_seed_{seed}"
    best_path = arguments.output_dir / f"{stem}.pt"
    progress_path = arguments.output_dir / f"{stem}_progress.pt"
    result_path = arguments.output_dir / f"{stem}_result.json"

    if arguments.resume and result_path.exists():
        completed = json.loads(result_path.read_text(encoding="utf-8"))
        if completed.get("training") != signature:
            raise ValueError(
                f"completed run {result_path} has different settings; "
                "choose another output directory"
            )
        print(f"seed {seed}: already complete; reusing {result_path.name}", flush=True)
        return completed

    seed_everything(seed)
    model = OutcomePredictionModel(
        embedding_rows=int(metadata["embedding_rows"]),
        level_mean=float(preprocessing["level_mean"]),
        level_standard_deviation=float(preprocessing["level_standard_deviation"]),
        head_hidden_dim=DEFAULT_OUTCOME_HEAD_HIDDEN_DIM,
        summary_dim=int(tensors["summary_features"].shape[-1]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    training_indices = np.flatnonzero(arrays["splits"] == 0)
    validation_indices = np.flatnonzero(arrays["splits"] == 1)
    test_indices = np.flatnonzero(arrays["splits"] == 2)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1

    if arguments.resume and progress_path.exists():
        checkpoint = torch.load(progress_path, map_location="cpu", weights_only=False)
        if checkpoint.get("training") != signature:
            raise ValueError(
                f"partial run {progress_path} has different settings; "
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
        print(f"seed {seed}: resuming at epoch {start_epoch}", flush=True)

    print(f"seed {seed}: training outcome model on {device}", flush=True)
    stopped_early = epochs_without_improvement >= arguments.patience
    for epoch in range(start_epoch, arguments.max_epochs + 1):
        if stopped_early:
            break
        model.train()
        order = training_indices.copy()
        np.random.default_rng(seed + epoch).shuffle(order)
        loss_sum = 0.0
        started = time.perf_counter()
        for start in range(0, len(order), arguments.batch_size):
            batch = select_outcome_batch(
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
        validation_loss = float(validation["overall"]["log_loss"])
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / len(order),
            "validation_log_loss": validation_loss,
            "validation_stay_log_loss": float(validation["stay"]["log_loss"]),
            "validation_switch_log_loss": float(validation["switch"]["log_loss"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        improved = validation_loss < best_loss - arguments.min_delta
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "model_state": cpu_state_dict(model),
                    "seed": seed,
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
            f"validation={validation_loss:.4f} "
            f"stay={record['validation_stay_log_loss']:.4f} "
            f"switch={record['validation_switch_log_loss']:.4f} "
            f"best={best_loss:.4f} time={record['seconds']:.1f}s",
            flush=True,
        )
        if epochs_without_improvement >= arguments.patience:
            stopped_early = True
            print("  early stopping", flush=True)

    if not best_path.exists():
        raise RuntimeError(f"seed {seed} did not produce a best checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    validation, _ = evaluate(
        model, tensors, validation_indices, arguments.batch_size, device
    )
    test, _ = evaluate(model, tensors, test_indices, arguments.batch_size, device)
    result = {
        "seed": seed,
        "name": "action_conditioned_outcome",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training": signature,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "validation": validation,
        "test": test,
        "history": history,
        "checkpoint": best_path.name,
    }
    atomic_json_dump(result, result_path)
    return result


def aggregate_outcome_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics for overall, stay, and switch factual predictions."""
    aggregate: dict[str, Any] = {}
    for split in ("validation", "test"):
        aggregate[split] = {}
        for group in ("overall", "stay", "switch"):
            aggregate[split][group] = {}
            metric_names = (
                name
                for name in runs[0][split][group]
                if name not in {"examples", "win_rate"}
            )
            for metric_name in metric_names:
                values = np.asarray(
                    [run[split][group][metric_name] for run in runs], dtype=float
                )
                aggregate[split][group][metric_name] = {
                    "mean": float(values.mean()),
                    "sample_standard_deviation": (
                        float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ),
                }
    return aggregate


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the factual action-conditioned battle-11 win baseline."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "switch_current_deck_count_ablation" / "arrays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "outcome_baseline",
    )
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="ignore partial and completed runs in the output directory",
    )
    parser.set_defaults(resume=True)
    arguments = parser.parse_args()
    if arguments.max_epochs < 1 or arguments.patience < 1 or arguments.batch_size < 1:
        parser.error("max epochs, patience, and batch size must be positive")
    if arguments.min_delta < 0.0:
        parser.error("min delta cannot be negative")
    if arguments.learning_rate <= 0.0 or arguments.weight_decay < 0.0:
        parser.error("learning rate must be positive and weight decay cannot be negative")
    if len(set(arguments.seeds)) != len(arguments.seeds):
        parser.error("seeds must be unique")
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
    if "next_wins" not in arrays:
        raise ValueError("outcome training requires the next_wins target array")
    if metadata.get("continuity") != "same_collection":
        raise ValueError("outcome training requires a same_collection cache")
    if metadata.get("summary_feature_set") != "current_deck_count":
        raise ValueError("outcome training requires the selected nine-feature cache")
    splits = np.asarray(arrays["splits"])
    validate_player_splits(np.asarray(arrays["player_indices"]), splits)
    if set(np.unique(splits)) != {0, 1, 2}:
        raise ValueError("outcome training requires train, validation, and test splits")

    arrays, preprocessing = prepare_arrays(arrays)
    tensors = outcome_tensors_on_device(arrays, device)
    runtime = environment_metadata(device)
    print(json.dumps({"data": metadata, "environment": runtime}, indent=2), flush=True)

    runs = []
    for seed in arguments.seeds:
        runs.append(
            train_seed(
                seed,
                arrays,
                tensors,
                metadata,
                preprocessing,
                arguments,
                device,
            )
        )
        report = {
            "data": metadata,
            "preprocessing": preprocessing,
            "environment": runtime,
            "runs": runs,
            "aggregate": aggregate_outcome_runs(runs),
        }
        atomic_json_dump(report, arguments.output_dir / "results.json")

    print("\nAGGREGATE FACTUAL TEST RESULTS", flush=True)
    for group in ("overall", "stay", "switch"):
        summary = report["aggregate"]["test"][group]["log_loss"]
        print(
            f"  {group:8s} log_loss={summary['mean']:.6f} "
            f"+/- {summary['sample_standard_deviation']:.6f}",
            flush=True,
        )
    print(f"written {arguments.output_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
