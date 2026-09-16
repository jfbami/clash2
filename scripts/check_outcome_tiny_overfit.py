"""Sanity-check whether the outcome network can memorize a tiny training set."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crdata.switch_dataset import load_switch_arrays
from models.outcome_model import OutcomePredictionModel
from scripts.train_outcome_baseline import (
    atomic_json_dump,
    outcome_tensors_on_device,
    prepare_arrays,
    resolve_device,
    seed_everything,
    validate_player_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = ((0, 0), (0, 1), (1, 0), (1, 1))


def select_training_examples(
    arrays: dict[str, np.ndarray], per_group: int, seed: int
) -> np.ndarray:
    """Choose one row per player in each observed action/outcome group."""
    splits = np.asarray(arrays["splits"])
    actions = np.asarray(arrays["labels"])
    outcomes = np.asarray(arrays["next_wins"])
    players = np.asarray(arrays["player_indices"])
    candidates = np.flatnonzero(splits == 0)
    np.random.default_rng(seed).shuffle(candidates)
    counts = {group: 0 for group in GROUPS}
    selected: list[int] = []
    used_players: set[int] = set()
    for index in candidates:
        group = (int(actions[index]), int(outcomes[index]))
        player = int(players[index])
        if group not in counts or counts[group] >= per_group or player in used_players:
            continue
        selected.append(int(index))
        used_players.add(player)
        counts[group] += 1
        if all(count == per_group for count in counts.values()):
            break
    if any(count != per_group for count in counts.values()):
        raise ValueError(f"not enough distinct training players for each group: {counts}")
    return np.asarray(selected, dtype=np.int64)


def validate_tiny_shapes(tensors: dict[str, torch.Tensor], count: int) -> None:
    """Make the batch, time, card, feature, and label contract explicit."""
    expected = {
        "cards": (count, 10, 8),
        "levels": (count, 10, 8),
        "battle_features": (count, 10, 7),
        "summary_features": (count, 9),
        "actions": (count,),
        "outcomes": (count,),
    }
    for name, shape in expected.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensors[name].shape)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether the existing outcome architecture can overfit 32 training rows."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "data/switch_current_deck_count_ablation/arrays",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/outcome_tiny_overfit/results.json",
    )
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--target-loss", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    if arguments.per_group < 1 or arguments.max_steps < 1:
        parser.error("per-group and max-steps must be positive")
    if arguments.target_loss <= 0.0 or arguments.learning_rate <= 0.0:
        parser.error("target-loss and learning-rate must be positive")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    device = resolve_device(arguments.device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    seed_everything(arguments.seed)

    arrays, metadata = load_switch_arrays(arguments.cache)
    if "next_wins" not in arrays:
        raise ValueError("cache must contain next_wins")
    if metadata.get("continuity") != "same_collection":
        raise ValueError("cache must use same_collection histories")
    if metadata.get("summary_feature_set") != "current_deck_count":
        raise ValueError("cache must contain the selected nine summary features")
    validate_player_splits(
        np.asarray(arrays["player_indices"]), np.asarray(arrays["splits"])
    )
    selected = select_training_examples(arrays, arguments.per_group, arguments.seed)
    train_indices = np.flatnonzero(arrays["splits"] == 0)
    prepared, preprocessing = prepare_arrays(arrays, train_indices)
    tiny_arrays = {
        name: prepared[name][selected]
        for name in (
            "cards", "levels", "battle_features", "summary_features", "labels", "next_wins"
        )
    }
    tensors = outcome_tensors_on_device(tiny_arrays, device)
    count = len(selected)
    validate_tiny_shapes(tensors, count)

    model = OutcomePredictionModel(
        embedding_rows=int(metadata["embedding_rows"]),
        level_mean=float(preprocessing["level_mean"]),
        level_standard_deviation=float(preprocessing["level_standard_deviation"]),
        summary_dim=9,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=0.0
    )
    criterion = nn.BCEWithLogitsLoss()
    inputs = (
        tensors["cards"], tensors["levels"], tensors["battle_features"],
        tensors["summary_features"], tensors["actions"],
    )
    targets = tensors["outcomes"]
    model.train()
    with torch.no_grad():
        initial_logits = model(*inputs)
        if initial_logits.shape != targets.shape:
            raise ValueError("win logits and outcomes must both have shape (batch,)")
        initial_loss = float(criterion(initial_logits, targets))
    trace: list[dict[str, float | int]] = [{"step": 0, "loss": initial_loss}]
    max_gradient_norm = 0.0

    for step in range(1, arguments.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*inputs)
        if logits.shape != targets.shape:
            raise ValueError("win logits and outcomes must both have shape (batch,)")
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
        if not np.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        max_gradient_norm = max(max_gradient_norm, gradient_norm)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == arguments.max_steps:
            trace.append({"step": step, "loss": float(loss.detach())})
            print(f"step {step:4d}: loss={float(loss.detach()):.6f}", flush=True)
        if float(loss.detach()) <= arguments.target_loss:
            break

    model.eval()
    with torch.inference_mode():
        final_logits = model(*inputs)
        final_loss = float(criterion(final_logits, targets))
        probabilities = torch.sigmoid(final_logits)
        accuracy = float(((probabilities >= 0.5) == targets.bool()).float().mean())
    if trace[-1]["step"] == step:
        trace[-1]["loss"] = final_loss
    else:
        trace.append({"step": step, "loss": final_loss})
    report = {
        "purpose": "training-only memorization sanity check, not generalization",
        "source_split": "train",
        "examples": count,
        "distinct_players": count,
        "examples_per_action_outcome_group": arguments.per_group,
        "seed": arguments.seed,
        "device": str(device),
        "dropout": 0.0,
        "weight_decay": 0.0,
        "gradient_clip_max_norm": 1.0,
        "max_gradient_norm_before_clipping": max_gradient_norm,
        "target_loss": arguments.target_loss,
        "steps_completed": step,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "target_reached": final_loss <= arguments.target_loss,
        "training_accuracy": accuracy,
        "trace": trace,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(report, arguments.output)
    print(json.dumps({key: value for key, value in report.items() if key != "trace"}, indent=2))
    print(f"written {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
