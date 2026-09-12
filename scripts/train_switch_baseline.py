"""Train a reproducible multi-seed next-switch baseline on CPU or GPU."""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crdata.battle_features import fit_battle_feature_standardization
from crdata.card_levels import fit_level_standardization
from crdata.summary_features import fit_summary_feature_standardization
from crdata.switch_dataset import load_switch_arrays
from models.switch_model import SwitchPredictionModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (17, 3407, 918273)
HEAD_HIDDEN_DIM = 128


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Return ten-bin absolute calibration error weighted by bin size."""
    edges = np.linspace(0.0, 1.0, 11)
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
    """Calculate discrimination, calibration, and threshold metrics."""
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    predictions = probabilities >= 0.5
    return {
        "log_loss": float(log_loss(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "calibration_error": expected_calibration_error(labels, probabilities),
        "accuracy": float(np.mean(predictions == labels)),
    }


def prepare_arrays(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit preprocessing on training players and transform model inputs."""
    training = np.flatnonzero(arrays["splits"] == 0)
    level_stats = fit_level_standardization(arrays["levels"][training])
    battle_stats = fit_battle_feature_standardization(arrays["battle_features"][training])
    summary_stats = fit_summary_feature_standardization(arrays["summary_features"][training])
    prepared = {
        **arrays,
        "battle_features": battle_stats.transform(arrays["battle_features"]),
        "summary_features": summary_stats.transform(arrays["summary_features"]),
    }
    preprocessing = {
        "level_mean": level_stats.mean,
        "level_standard_deviation": level_stats.standard_deviation,
        "battle_feature_means": list(battle_stats.means),
        "battle_feature_standard_deviations": list(battle_stats.standard_deviations),
        "summary_feature_means": list(summary_stats.means),
        "summary_feature_standard_deviations": list(summary_stats.standard_deviations),
    }
    return prepared, preprocessing


def tensors_on_device(
    arrays: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    """Materialize the compact training arrays once on the selected device."""
    numpy_arrays = {
        "cards": np.asarray(arrays["cards"]).astype(np.int64),
        "levels": np.asarray(arrays["levels"]).astype(np.float32),
        "battle_features": np.asarray(arrays["battle_features"], dtype=np.float32),
        "summary_features": np.asarray(arrays["summary_features"], dtype=np.float32),
        "labels": np.asarray(arrays["labels"]).astype(np.float32),
    }
    return {
        name: torch.from_numpy(values).to(device)
        for name, values in numpy_arrays.items()
    }


def select_batch(
    tensors: dict[str, torch.Tensor], indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Select one batch from tensors already resident on the training device."""
    index = torch.as_tensor(indices, dtype=torch.long, device=device)
    return (
        tensors["cards"].index_select(0, index),
        tensors["levels"].index_select(0, index),
        tensors["battle_features"].index_select(0, index),
        tensors["summary_features"].index_select(0, index),
        tensors["labels"].index_select(0, index),
    )


def evaluate(
    model: SwitchPredictionModel,
    tensors: dict[str, torch.Tensor],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate a split without retaining activations on the accelerator."""
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = select_batch(tensors, indices[start:start + batch_size], device)
            outputs.append(model(*batch[:-1]).detach().cpu())
    logits = torch.cat(outputs).numpy()
    labels = tensors["labels"].index_select(
        0, torch.as_tensor(indices, dtype=torch.long, device=device)
    ).detach().cpu().numpy()
    return metrics(labels, logits), logits


def seed_everything(seed: int) -> None:
    """Seed initialization, dropout, and all available CUDA generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy model parameters to CPU for portable checkpoints."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def atomic_torch_save(value: Any, path: Path) -> None:
    """Write a checkpoint atomically so an interruption cannot corrupt it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json_dump(value: Any, path: Path) -> None:
    """Write indented JSON atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move restored optimizer buffers to the active device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    """Restore dropout generators so resumed training follows the same run."""
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_states = checkpoint.get("cuda_rng_states")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def training_signature(arguments: argparse.Namespace, seed: int) -> dict[str, Any]:
    """Return settings that must match before an interrupted run can resume."""
    return {
        "seed": seed,
        "max_epochs": arguments.max_epochs,
        "patience": arguments.patience,
        "min_delta": arguments.min_delta,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "weight_decay": arguments.weight_decay,
        "head_hidden_dim": HEAD_HIDDEN_DIM,
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
    """Train one seed, checkpointing every epoch and restoring the best model."""
    signature = training_signature(arguments, seed)
    stem = f"expanded_128_seed_{seed}"
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
    model = SwitchPredictionModel(
        embedding_rows=int(metadata["embedding_rows"]),
        level_mean=float(preprocessing["level_mean"]),
        level_standard_deviation=float(preprocessing["level_standard_deviation"]),
        head_hidden_dim=HEAD_HIDDEN_DIM,
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

    print(f"seed {seed}: training on {device}", flush=True)
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
            f"validation={record['validation_log_loss']:.4f} "
            f"best={best_loss:.4f} time={record['seconds']:.1f}s",
            flush=True,
        )
        if epochs_without_improvement >= arguments.patience:
            stopped_early = True
            print(
                f"  early stopping after {arguments.patience} epochs without improvement",
                flush=True,
            )
            break

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
        "name": "expanded_128",
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


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-seed metrics with sample standard deviations."""
    aggregate: dict[str, Any] = {}
    for split in ("validation", "test"):
        aggregate[split] = {}
        for metric_name in runs[0][split]:
            values = np.asarray([run[split][metric_name] for run in runs], dtype=float)
            aggregate[split][metric_name] = {
                "mean": float(values.mean()),
                "sample_standard_deviation": (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                ),
            }
    epochs = np.asarray([run["best_epoch"] for run in runs], dtype=float)
    aggregate["best_epoch"] = {
        "mean": float(epochs.mean()),
        "sample_standard_deviation": float(epochs.std(ddof=1)) if len(epochs) > 1 else 0.0,
    }
    return aggregate


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic accelerator selection."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def environment_metadata(device: torch.device) -> dict[str, Any]:
    """Record enough runtime detail to interpret accelerator results."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "cuda": torch.version.cuda,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the unchanged expanded switch model across reproducible seeds."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "data" / "switch_continuity_same_collection" / "arrays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "switch_baseline_expanded_128",
    )
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
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
    if metadata.get("continuity") != "same_collection":
        raise ValueError("baseline training requires a same_collection cache")
    arrays, preprocessing = prepare_arrays(arrays)
    tensors = tensors_on_device(arrays, device)
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
            "aggregate": aggregate_runs(runs),
        }
        atomic_json_dump(report, arguments.output_dir / "results.json")

    print("\nAGGREGATE TEST RESULTS", flush=True)
    for name, summary in report["aggregate"]["test"].items():
        print(
            f"  {name:18s} {summary['mean']:.6f} "
            f"+/- {summary['sample_standard_deviation']:.6f}",
            flush=True,
        )
    print(f"written {arguments.output_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
