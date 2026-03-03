"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger, evaluate_policy
import matplotlib.pyplot as plt

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "Flow"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    model = torch.compile(model) 
    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    step = 0
    train_steps, train_losses = [], []
    eval_steps, eval_rewards = [], []

    for epoch in range(config.num_epochs):
        for batch in loader:
            states_batch = batch[0].to(device)  
            actions_batch = batch[1].to(device) 

            optimizer.zero_grad()
            loss = model.compute_loss(states_batch, actions_batch)
            loss.backward()
            optimizer.step()

            if step % config.log_interval == 0:
                train_steps.append(step)
                train_losses.append(loss.item())
                wandb.log({"train/loss": loss.item()}, step=step)
                
            if step % config.eval_interval == 0:
                evaluate_policy(
                    model,
                    normalizer,
                    device,
                    config.chunk_size,
                    config.video_size,
                    config.num_video_episodes,
                    config.flow_num_steps,
                    step,
                    logger,
                )
                mean_reward = logger.rows[-1].get("eval/mean_reward")
                if mean_reward is not None:
                    eval_steps.append(step)
                    eval_rewards.append(mean_reward)

            step += 1


     # Plot training curves
    log_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_steps, train_losses, linewidth=1)
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{config.policy_type} Training Loss")
    axes[0].grid(True, alpha=0.3)

    if eval_rewards:
        axes[1].plot(eval_steps, eval_rewards, marker="o", linewidth=1)
        axes[1].set_xlabel("Training Steps")
        axes[1].set_ylabel("Mean Reward")
        axes[1].set_title("Evaluation Reward")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_title("No eval rewards returned by evaluate_policy")

    plt.tight_layout()
    plot_path = log_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {plot_path}")
    
    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
