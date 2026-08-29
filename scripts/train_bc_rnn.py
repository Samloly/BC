import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mybc.dataset import (
    SequenceRobomimicDataset,
)
from mybc.normalizer import (
    ObservationNormalizer,
    compute_obs_statistics,
)
from mybc.policy import RNNPolicy
from mybc.trainer import (
    train_rnn_one_epoch,
    validate_rnn,
)


DEFAULT_OBS_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a GRU behavioral cloning "
            "policy on a robomimic dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/lift/ph/low_dim_v15.hdf5"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/best_bc_rnn.pth"
        ),
    )

    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path(
            "results/bc_rnn_loss_history.json"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
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
        "--sequence-length",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
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
        "--no-pad-sequence",
        action="store_true",
        help=(
            "Disable sequence-end padding. "
            "Only complete sequences will "
            "be used."
        ),
    )

    parser.add_argument(
        "--use-pad-mask",
        action="store_true",
        help=(
            "Exclude padded timesteps from "
            "the training and validation loss."
        ),
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help=(
            "Stop if validation loss does not "
            "improve for this many epochs. "
            "Use 0 to disable early stopping."
        ),
    )

    return parser.parse_args()


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be greater than zero"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero"
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be positive"
        )

    if args.sequence_length <= 0:
        raise ValueError(
            "--sequence-length must be positive"
        )

    if args.hidden_size <= 0:
        raise ValueError(
            "--hidden-size must be positive"
        )

    if args.num_layers <= 0:
        raise ValueError(
            "--num-layers must be positive"
        )

    if args.dropout < 0:
        raise ValueError(
            "--dropout must be non-negative"
        )

    if (
        args.num_layers == 1
        and args.dropout > 0
    ):
        print(
            "Warning: GRU dropout is only "
            "applied between recurrent layers. "
            "It has no effect when "
            "num_layers=1."
        )

    if args.patience < 0:
        raise ValueError(
            "--patience must be non-negative"
        )


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
    sequence_length,
    hidden_size,
    num_layers,
    dropout,
    pad_sequence,
    use_pad_mask,
    epoch,
    validation_loss,
    dataset_path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_type": "bc_rnn",
        "rnn_type": "gru",
        "policy_state_dict": (
            policy.state_dict()
        ),
        "normalizer_state_dict": (
            normalizer.state_dict()
        ),
        "observation_statistics": (
            statistics
        ),
        "obs_keys": list(obs_keys),
        "obs_shapes": obs_shapes,
        "action_dim": int(action_dim),
        "sequence_length": int(
            sequence_length
        ),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "pad_sequence": bool(
            pad_sequence
        ),
        "use_pad_mask": bool(
            use_pad_mask
        ),
        "epoch": int(epoch),
        "validation_loss": float(
            validation_loss
        ),
        "dataset_path": str(
            dataset_path
        ),
    }

    torch.save(
        checkpoint,
        output_path,
    )


def print_batch_shapes(batch):
    print(
        "Action batch:",
        tuple(batch["actions"].shape),
    )

    for key, value in batch[
        "obs"
    ].items():
        print(
            f"Observation {key}:",
            tuple(value.shape),
        )

    if "pad_mask" in batch:
        print(
            "Pad mask:",
            tuple(
                batch["pad_mask"].shape
            ),
        )


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{args.dataset}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    obs_keys = DEFAULT_OBS_KEYS

    pad_sequence = (
        not args.no_pad_sequence
    )

    # A mask only needs to be returned when
    # masked loss has been requested.
    get_pad_mask = args.use_pad_mask

    print("=" * 60)
    print("Device:", device)
    print("Dataset:", args.dataset)
    print("Observation keys:", obs_keys)
    print(
        "Sequence length:",
        args.sequence_length,
    )
    print(
        "Pad sequence:",
        pad_sequence,
    )
    print(
        "Use pad mask:",
        args.use_pad_mask,
    )
    print(
        "GRU hidden size:",
        args.hidden_size,
    )
    print(
        "GRU layers:",
        args.num_layers,
    )
    print("=" * 60)

    train_dataset = None
    valid_dataset = None

    try:
        train_dataset = (
            SequenceRobomimicDataset(
                dataset_path=args.dataset,
                split="train",
                obs_keys=obs_keys,
                sequence_length=(
                    args.sequence_length
                ),
                pad_sequence=pad_sequence,
                get_pad_mask=get_pad_mask,
            )
        )

        valid_dataset = (
            SequenceRobomimicDataset(
                dataset_path=args.dataset,
                split="valid",
                obs_keys=obs_keys,
                sequence_length=(
                    args.sequence_length
                ),
                pad_sequence=pad_sequence,
                get_pad_mask=get_pad_mask,
            )
        )

        obs_shapes = (
            train_dataset.get_obs_shape()
        )

        action_dim = (
            train_dataset.get_action_dim()
        )

        # Statistics are calculated from raw
        # training observations, not padded
        # sequences. The same statistics are
        # reused by MLP and RNN.
        statistics = (
            compute_obs_statistics(
                dataset_path=args.dataset,
                split="train",
                obs_keys=obs_keys,
            )
        )

        normalizer = (
            ObservationNormalizer(
                statistics=statistics,
                obs_keys=obs_keys,
            ).to(device)
        )

        policy = RNNPolicy(
            obs_shapes=obs_shapes,
            obs_keys=obs_keys,
            action_dim=action_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=args.learning_rate,
            weight_decay=(
                args.weight_decay
            ),
        )

        pin_memory = (
            device.type == "cuda"
        )

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

        print(
            "Observation shapes:",
            obs_shapes,
        )
        print(
            "Observation dimension:",
            policy.observation_dim,
        )
        print(
            "Action dimension:",
            action_dim,
        )
        print(
            "Training sequences:",
            len(train_dataset),
        )
        print(
            "Validation sequences:",
            len(valid_dataset),
        )
        print(policy)

        # Run a shape check before training.
        example_batch = next(
            iter(train_loader)
        )

        print_batch_shapes(
            example_batch
        )

        expected_action_shape = (
            example_batch["actions"].shape[
                0
            ],
            args.sequence_length,
            action_dim,
        )

        if (
            tuple(
                example_batch[
                    "actions"
                ].shape
            )
            != expected_action_shape
        ):
            raise RuntimeError(
                "Unexpected action batch "
                f"shape: "
                f"{tuple(example_batch['actions'].shape)}; "
                f"expected "
                f"{expected_action_shape}"
            )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.history_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        best_validation_loss = float(
            "inf"
        )

        epochs_without_improvement = 0
        history = []

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            train_metrics = (
                train_rnn_one_epoch(
                    policy=policy,
                    normalizer=normalizer,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    max_grad_norm=(
                        args.max_grad_norm
                    ),
                    use_pad_mask=(
                        args.use_pad_mask
                    ),
                )
            )

            valid_metrics = validate_rnn(
                policy=policy,
                normalizer=normalizer,
                data_loader=valid_loader,
                device=device,
                use_pad_mask=(
                    args.use_pad_mask
                ),
            )

            history_item = {
                "epoch": epoch,
                "train_loss": float(
                    train_metrics["loss"]
                ),
                "validation_loss": float(
                    valid_metrics["loss"]
                ),
            }

            if "grad_norm" in train_metrics:
                history_item[
                    "grad_norm"
                ] = float(
                    train_metrics[
                        "grad_norm"
                    ]
                )

            history.append(
                history_item
            )

            print(
                f"Epoch "
                f"{epoch:03d}/"
                f"{args.epochs} | "
                f"train_loss="
                f"{train_metrics['loss']:.6f} | "
                f"valid_loss="
                f"{valid_metrics['loss']:.6f}",
                end="",
            )

            if "grad_norm" in train_metrics:
                print(
                    f" | grad_norm="
                    f"{train_metrics['grad_norm']:.4f}",
                    end="",
                )

            print()

            improved = (
                valid_metrics["loss"]
                < best_validation_loss
            )

            if improved:
                best_validation_loss = (
                    valid_metrics["loss"]
                )

                epochs_without_improvement = 0

                save_checkpoint(
                    output_path=args.output,
                    policy=policy,
                    normalizer=normalizer,
                    statistics=statistics,
                    obs_keys=obs_keys,
                    obs_shapes=obs_shapes,
                    action_dim=action_dim,
                    sequence_length=(
                        args.sequence_length
                    ),
                    hidden_size=(
                        args.hidden_size
                    ),
                    num_layers=(
                        args.num_layers
                    ),
                    dropout=args.dropout,
                    pad_sequence=(
                        pad_sequence
                    ),
                    use_pad_mask=(
                        args.use_pad_mask
                    ),
                    epoch=epoch,
                    validation_loss=(
                        best_validation_loss
                    ),
                    dataset_path=(
                        args.dataset
                    ),
                )

                print(
                    "Saved best checkpoint:",
                    args.output,
                )

            else:
                epochs_without_improvement += 1

            with (
                args.history_output.open(
                    "w",
                    encoding="utf-8",
                )
            ) as file:
                json.dump(
                    history,
                    file,
                    indent=2,
                )

            if (
                args.patience > 0
                and epochs_without_improvement
                >= args.patience
            ):
                print(
                    "Early stopping: "
                    f"validation loss did not "
                    f"improve for "
                    f"{args.patience} epochs."
                )
                break

        print("=" * 60)
        print(
            "Best validation loss:",
            best_validation_loss,
        )
        print(
            "Best checkpoint:",
            args.output,
        )
        print(
            "Training history:",
            args.history_output,
        )
        print("=" * 60)

    finally:
        if train_dataset is not None:
            train_dataset.close()

        if valid_dataset is not None:
            valid_dataset.close()


if __name__ == "__main__":
    main()
