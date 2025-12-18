"""Simple utilities for generating synthetic regression data and saving CSV files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class RegressionParameters:
    #Holds true regression weights and bias for reproducible data generation.

    weights: Array
    bias: float

    @property
    def input_dim(self) -> int:
        return self.weights.shape[0]


DEFAULT_PARAMS = {
    1: RegressionParameters(weights=np.array([1.5]), bias=0.4),
    2: RegressionParameters(weights=np.array([1.2, -0.8]), bias=-0.3),
}


def generate_inputs(n_samples: int, input_dim: int, rng: np.random.Generator) -> Array:
    #Draw uniformly distributed inputs in [-3, 3].

    return rng.uniform(-3.0, 3.0, size=(n_samples, input_dim))


def evaluate_regression(X: Array, params: RegressionParameters) -> Array:
    #Return noiseless regression output for inputs X.

    if X.ndim != 2 or X.shape[1] != params.input_dim:
        raise ValueError("X must have shape (n_samples, %d)" % params.input_dim)
    return X @ params.weights + params.bias


def generate_dataset(
    n_samples: int,
    input_dim: int,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[Array, Array, Array]:
    #Generate inputs, noisy outputs, and noiseless targets.

    if input_dim not in DEFAULT_PARAMS:
        raise ValueError("Only input dimensions %s are supported." % sorted(DEFAULT_PARAMS))
    params = DEFAULT_PARAMS[input_dim]
    X = generate_inputs(n_samples, input_dim, rng)
    y_clean = evaluate_regression(X, params)
    noise = rng.normal(scale=noise_std, size=n_samples)
    y_noisy = y_clean + noise
    return X, y_noisy


def save_csv(path: Path, X: Array, y: Array) -> None:
    #Save generated dataset into CSV with columns x*, y.

    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")

    columns = [f"x{i+1}" for i in range(X.shape[1])]
    columns.extend(["y"])
    header = ",".join(columns)
    data = np.column_stack([X, y])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic regression CSV data.")
    parser.add_argument("output", type=Path, help="Path to output CSV file.")
    parser.add_argument("--samples", type=int, default=200, help="Number of data points to generate.")
    parser.add_argument(
        "--input-dim",
        type=int,
        choices=tuple(DEFAULT_PARAMS.keys()),
        default=1,
        help="Input dimensionality (1 or 2).",
    )
    parser.add_argument("--noise-std", type=float, default=0.1, help="Std. dev. of Gaussian noise added to y.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    X, y = generate_dataset(args.samples, args.input_dim, args.noise_std, rng)
    save_csv(args.output, X, y)
    print("Saved dataset to", args.output)
    print("True parameters:", DEFAULT_PARAMS[args.input_dim])


if __name__ == "__main__":
    main()
