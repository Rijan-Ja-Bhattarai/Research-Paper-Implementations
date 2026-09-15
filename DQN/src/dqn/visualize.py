from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return x positions and a rolling mean, including short training runs."""
    window = max(1, min(window, len(values)))
    means = np.convolve(values, np.ones(window) / window, mode="valid")
    positions = np.arange(window - 1, len(values))
    return positions, means


def plot_training(
    episode_returns: Sequence[float],
    td_errors: Sequence[float],
    output_path: str | Path = "training_progress.png",
    window: int = 500,
    show: bool = False,
) -> None:
    """Plot agent performance and learning stability, then save the figure."""
    returns = np.asarray(episode_returns, dtype=float)
    errors = np.abs(np.asarray(td_errors, dtype=float))
    if returns.size == 0 or errors.size == 0:
        raise ValueError("Train the agent before plotting its metrics.")

    reward_x, smooth_returns = _rolling_mean(returns, window)
    error_x, smooth_errors = _rolling_mean(errors, window)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)

    axes[0].plot(returns, color="tab:blue", alpha=0.12, linewidth=0.7)
    axes[0].plot(reward_x, smooth_returns, color="tab:blue", linewidth=2)
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set(title="Episode return", ylabel="Reward")

    axes[1].plot(errors, color="tab:orange", alpha=0.08, linewidth=0.7)
    axes[1].plot(error_x, smooth_errors, color="tab:orange", linewidth=2)
    axes[1].set(
        title="Absolute temporal-difference error",
        xlabel="Environment step",
        ylabel="|TD error|",
    )

    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Training progress ({window}-sample rolling averages)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
