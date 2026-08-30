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
from mybc.policy import TransformerPolicy
from mybc.trainer import (
    train_transformer_one_epoch,
    validate_transformer,
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
            "Train a causal Transformer "
            "behavioral cloning policy."
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
            "checkpoints/"
            "best_bc_transformer.pth"
        ),
    )

    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path(
            "results/"
            "bc_transformer_loss_history.json"
        ),
    )

    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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
        "--patience",
        type=int,
        default=10,
    )

    # 序列参数
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--no-pad-sequence",
        action="store_true",
    )

    parser.add_argument(
        "--no-pad-mask",
        action="store_true",
        help=(
            "Do not mask padded timesteps. "
            "Normally this should not be used."
        ),
    )

    # Transformer参数
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
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
    args,
    epoch,
    validation_loss,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_type": "bc_transformer",

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
            args.sequence_length
        ),
        "d_model": int(args.d_model),
        "nhead": int(args.nhead),
        "num_layers": int(
            args.num_layers
        ),
        "dim_feedforward": int(
            args.dim_feedforward
        ),
        "dropout": float(args.dropout),

        "pad_sequence": bool(
            not args.no_pad_sequence
        ),
        "use_pad_mask": bool(
            not args.no_pad_mask
        ),

        "epoch": int(epoch),
        "validation_loss": float(
            validation_loss
        ),
        "dataset_path": str(
            args.dataset
        ),
    }

    torch.save(
        checkpoint,
        output_path,
    )

def print_batch_shapes(batch):
    print(
        "Actions:",
        tuple(batch["actions"].shape),
    )

    for key, value in batch["obs"].items():
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
    # validate_args(args)
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

    use_pad_mask = (
        not args.no_pad_mask
    )

    # 只有进行 padding 时才需要 mask
    get_pad_mask = (
        pad_sequence and use_pad_mask
    )

    print("=" * 60)
    print("Device:", device)
    print("Dataset:", args.dataset)
    print(
        "Sequence length:",
        args.sequence_length,
    )
    print("d_model:", args.d_model)
    print("Attention heads:", args.nhead)
    print(
        "Transformer layers:",
        args.num_layers,
    )
    print(
        "Feedforward dimension:",
        args.dim_feedforward,
    )
    print("Dropout:", args.dropout)
    print("Pad sequence:", pad_sequence)
    print("Use pad mask:", get_pad_mask)
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

        action_dim = int(
            train_dataset.get_action_dim()
        )

        statistics = compute_obs_statistics(
            dataset_path=args.dataset,
            split="train",
            obs_keys=obs_keys,
        )

        normalizer = ObservationNormalizer(
            statistics=statistics,
            obs_keys=obs_keys,
        ).to(device)

        policy = TransformerPolicy(
            obs_shapes=obs_shapes,
            obs_keys=obs_keys,
            action_dim=action_dim,
            sequence_length=(
                args.sequence_length
            ),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=(
                args.dim_feedforward
            ),
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
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

        # 训练前形状检查
        example_batch = next(
            iter(train_loader)
        )

        print_batch_shapes(
            example_batch
        )

        expected_action_shape = (
            example_batch["actions"].shape[0],
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
                "Unexpected action shape: "
                f"{tuple(example_batch['actions'].shape)}; "
                f"expected {expected_action_shape}"
            )

        # Policy前向传播检查
        example_obs = {
            key: value.to(device)
            for key, value
            in example_batch["obs"].items()
        }

        example_obs = normalizer(
            example_obs
        )

        example_padding_mask = None

        if get_pad_mask:
            example_valid_mask = (
                example_batch[
                    "pad_mask"
                ].to(device)
            )

            example_padding_mask = (
                ~example_valid_mask
                .squeeze(-1)
                .bool()
            )

        with torch.no_grad():
            example_prediction = policy(
                example_obs,
                padding_mask=(
                    example_padding_mask
                ),
            )

        print(
            "Prediction:",
            tuple(
                example_prediction.shape
            ),
        )

        if (
            tuple(example_prediction.shape)
            != expected_action_shape
        ):
            raise RuntimeError(
                "Unexpected policy output: "
                f"{tuple(example_prediction.shape)}; "
                f"expected {expected_action_shape}"
            )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.history_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        history = []

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            train_metrics = (
                train_transformer_one_epoch(
                    policy=policy,
                    normalizer=normalizer,
                    data_loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    max_grad_norm=(
                        args.max_grad_norm
                    ),
                    use_pad_mask=(
                        get_pad_mask
                    ),
                )
            )

            valid_metrics = (
                validate_transformer(
                    policy=policy,
                    normalizer=normalizer,
                    data_loader=valid_loader,
                    device=device,
                    use_pad_mask=(
                        get_pad_mask
                    ),
                )
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

            history.append(history_item)

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
                    args=args,
                    epoch=epoch,
                    validation_loss=(
                        best_validation_loss
                    ),
                )

                print(
                    "Saved best checkpoint:",
                    args.output,
                )

            else:
                epochs_without_improvement += 1

            with args.history_output.open(
                "w",
                encoding="utf-8",
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
                    "Early stopping: no "
                    "validation improvement "
                    f"for {args.patience} epochs."
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