"""Render the implemented switch model with VisualTorch."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import visualtorch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.switch_model import SwitchPredictionModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the implemented switch-model architecture."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "architecture_assets" / "switch-model-visualtorch.png",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    model = SwitchPredictionModel(
        embedding_rows=124,
        level_mean=0.0,
        level_standard_deviation=1.0,
        head_hidden_dim=128,
        summary_dim=9,
    ).eval()
    image = visualtorch.render(
        model,
        (
            (1, 10, 8),  # card indices
            (1, 10, 8),  # displayed card levels
            (1, 10, 7),  # preprocessed battle features
            (1, 9),  # long-term player summaries
        ),
        style="graph",
        input_dtype=(torch.long, torch.float32, torch.float32, torch.float32),
        show_neurons=False,
        show_dimension=True,
        show_arrows=True,
        legend=True,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(arguments.output)
    print(f"written {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
