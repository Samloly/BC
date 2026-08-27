import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mybc.dataset import RobomimicDataset
from mybc.normalizer import (
    ObservationNormalizer,
    compute_observation_statistics,
)
from mybc.policy import MLPPolicy
from mybc.trainer import (
    train_one_epoch,
    validate,
)


DEFAULT_OBS_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/lift_ph_lowdim_v15.hdf5"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/best_bc_mlp.pth"
        ),
    )

    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path(
            "results/loss_history.json"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    output_path,
    policy,
    normalizer,
    statistics,
    obs_keys,
    obs_shapes,
    action_dim,
    hidden_dims,
    epoch,
    validation_loss,
    dataset_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "policy_state_dict": (
            policy.state_dict()
        ),
        "normalizer_state_dict": (
            normalizer.state_dict()
        ),
        "observation_statistics": statistics,
        "obs_keys": list(obs_keys),
        "obs_shapes": obs_shapes,
        "action_dim": action_dim,
        "hidden_dims": list(hidden_dims),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "dataset_path": str(dataset_path),
    }

    torch.save(
        checkpoint,
        output_path,
    )


def main():
    args = parse_args()

    set_seed(args.seed)

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    obs_keys = DEFAULT_OBS_KEYS
    hidden_dims = (256, 256)

    print("Device:", device)
    print("Dataset:", args.dataset)

    train_dataset = RobomimicDataset(
        dataset_path=args.dataset,
        split="train",
        obs_keys=obs_keys,
    )

    valid_dataset = RobomimicDataset(
        dataset_path=args.dataset,
        split="valid",
        obs_keys=obs_keys,
    )
    obs_shapes = (
        train_dataset.get_obs_shape()
    )

    action_dim = (
        train_dataset.get_action_dim()
    )

    statistics = (
        compute_observation_statistics(
            dataset_path=args.dataset,
            split="train",
            obs_keys=obs_keys,
        )
    )

    normalizer = ObservationNormalizer(
        statistics=statistics,
        obs_keys=obs_keys,
    ).to(device)

    policy = MLPPolicy(
        obs_shapes=obs_shapes,
        obs_keys=obs_keys,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    print("Observation keys:", obs_keys)
    print("Observation shapes:", obs_shapes)
    print("Action dimension:", action_dim)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(valid_dataset))
    print(policy)

    # Shape sanity check
    example_batch = next(
        iter(train_loader)
    )

    print(
        "Action batch:",
        tuple(example_batch["actions"].shape),
    )

    for key, value in example_batch[
        "obs"
    ].items():
        print(
            f"Observation {key}:",
            tuple(value.shape),
        )

    best_validation_loss = float("inf")
    history = []

    args.history_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            policy=policy,
            normalizer=normalizer,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_grad_norm=args.max_grad_norm,
        )

        valid_metrics = validate(
            policy=policy,
            normalizer=normalizer,
            data_loader=valid_loader,
            device=device,
        )

        history_item = {
            "epoch": epoch,
            "train_loss": train_metrics[
                "loss"
            ],
            "validation_loss": valid_metrics[
                "loss"
            ],
        }

        if "grad_norm" in train_metrics:
            history_item["grad_norm"] = (
                train_metrics["grad_norm"]
            )

        history.append(history_item)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"valid_loss={valid_metrics['loss']:.6f}"
        )

        if (
            valid_metrics["loss"]
            < best_validation_loss
        ):
            best_validation_loss = (
                valid_metrics["loss"]
            )

            save_checkpoint(
                output_path=args.output,
                policy=policy,
                normalizer=normalizer,
                statistics=statistics,
                obs_keys=obs_keys,
                obs_shapes=obs_shapes,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                epoch=epoch,
                validation_loss=best_validation_loss,
                dataset_path=args.dataset,
            )

            print(
                "Saved best checkpoint:",
                args.output,
            )

        with args.history_output.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                history,
                file,
                indent=2,
            )

    train_dataset.close()
    valid_dataset.close()

    print(
        "Best validation loss:",
        best_validation_loss,
    )


if __name__ == "__main__":
    main()