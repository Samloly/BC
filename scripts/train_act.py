import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mybc.act import ACTPolicy
from mybc.dataset import (
    ACTRobomimicDataset,
)
from mybc.normalizer import (
    ActionNormalizer,
    ObservationNormalizer,
    compute_action_statistics,
    compute_obs_statistics,
)
from mybc.trainer import (
    batch_to_device,
)

LOW_DIM_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)

RGB_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train deterministic ACT on a "
            "robomimic image dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/"
            "best_deterministic_act.pth"
        ),
    )

    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path(
            "results/"
            "deterministic_act_history.json"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=1e-5,
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
        "--chunk-size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help=(
            "Repeatedly train on one fixed "
            "batch to verify the pipeline."
        ),
    )

    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=1000,
    )

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_masked_l1_loss(predicted_actions,target_actions,is_pad):
    expected_mask_shape = (target_actions.shape[0],target_actions.shape[1])
    absolute_error = (predicted_actions-target_actions).abs()
    valid_mask = (
        ~is_pad
    ).unsqueeze(-1)

    expanded_mask = (
        valid_mask.expand_as(
            absolute_error
        )
    )

    loss_weight = (
        expanded_mask.sum()
    )

    loss = (
        absolute_error
        .masked_select(
            expanded_mask
        )
        .sum()
        / loss_weight.clamp_min(1)
    )

    return (
        loss,
        int(
            loss_weight.detach().item()
        ),
    )


def prepare_batch(
    raw_batch,
    device,
    observation_normalizer,
    action_normalizer,
):
    batch = batch_to_device(
        raw_batch,
        device,
        non_blocking=True,
    )

    normalized_obs = (
        observation_normalizer(
            batch["obs"]
        )
    )

    target_actions = (
        action_normalizer.normalize(
            batch["actions"]
        )
    )

    # padding经过归一化后不一定为0，
    # 因此重新清零
    target_actions = (
        target_actions.masked_fill(
            batch["is_pad"].unsqueeze(-1),
            0.0,
        )
    )

    return (
        normalized_obs,
        target_actions,
        batch["is_pad"],
    )


def build_optimizer(
    policy,
    learning_rate,
    backbone_learning_rate,
    weight_decay,
):
    backbone_parameters = []
    other_parameters = []

    backbone_prefix = (
        "observation_encoder."
        "visual_encoder.backbone."
    )

    for name, parameter in (
        policy.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if name.startswith(
            backbone_prefix
        ):
            backbone_parameters.append(
                parameter
            )
        else:
            other_parameters.append(
                parameter
            )

    parameter_groups = [
        {
            "params": other_parameters,
            "lr": learning_rate,
        }
    ]

    if backbone_parameters:
        parameter_groups.append(
            {
                "params": (
                    backbone_parameters
                ),
                "lr": (
                    backbone_learning_rate
                ),
            }
        )

    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=weight_decay,
    )


def train_one_epoch(
    policy,
    observation_normalizer,
    action_normalizer,
    data_loader,
    optimizer,
    device,
    max_grad_norm,
):
    policy.train()
    observation_normalizer.eval()
    action_normalizer.eval()

    total_loss = 0.0
    total_weight = 0
    total_grad_norm = 0.0
    number_of_updates = 0

    for raw_batch in data_loader:
        (
            normalized_obs,
            target_actions,
            is_pad,
        ) = prepare_batch(
            raw_batch=raw_batch,
            device=device,
            observation_normalizer=(
                observation_normalizer
            ),
            action_normalizer=(
                action_normalizer
            ),
        )

        predicted_actions = policy(
            normalized_obs
        )

        loss, loss_weight = (
            compute_masked_l1_loss(
                predicted_actions=(
                    predicted_actions
                ),
                target_actions=(
                    target_actions
                ),
                is_pad=is_pad,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        if max_grad_norm is not None:
            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    policy.parameters(),
                    max_grad_norm,
                )
            )

            total_grad_norm += float(
                grad_norm
            )

        optimizer.step()

        total_loss += (
            float(loss.detach())
            * loss_weight
        )

        total_weight += loss_weight
        number_of_updates += 1

    metrics = {
        "loss": (
            total_loss
            / max(total_weight, 1)
        ),
        "num_updates": (
            number_of_updates
        ),
    }

    if max_grad_norm is not None:
        metrics["grad_norm"] = (
            total_grad_norm
            / max(number_of_updates, 1)
        )

    return metrics


@torch.no_grad()
def validate(
    policy,
    observation_normalizer,
    action_normalizer,
    data_loader,
    device,
):
    policy.eval()
    observation_normalizer.eval()
    action_normalizer.eval()

    total_loss = 0.0
    total_weight = 0
    number_of_batches = 0

    for raw_batch in data_loader:
        (
            normalized_obs,
            target_actions,
            is_pad,
        ) = prepare_batch(
            raw_batch=raw_batch,
            device=device,
            observation_normalizer=(
                observation_normalizer
            ),
            action_normalizer=(
                action_normalizer
            ),
        )

        predicted_actions = policy(
            normalized_obs
        )

        loss, loss_weight = (
            compute_masked_l1_loss(
                predicted_actions=(
                    predicted_actions
                ),
                target_actions=(
                    target_actions
                ),
                is_pad=is_pad,
            )
        )

        total_loss += (
            float(loss)
            * loss_weight
        )

        total_weight += loss_weight
        number_of_batches += 1

    return {
        "loss": (
            total_loss
            / max(total_weight, 1)
        ),
        "num_batches": (
            number_of_batches
        ),
    }


def overfit_one_batch(
    policy,
    observation_normalizer,
    action_normalizer,
    raw_batch,
    optimizer,
    device,
    max_grad_norm,
    steps,
):
    policy.train()
    observation_normalizer.eval()
    action_normalizer.eval()

    (
        normalized_obs,
        target_actions,
        is_pad,
    ) = prepare_batch(
        raw_batch=raw_batch,
        device=device,
        observation_normalizer=(
            observation_normalizer
        ),
        action_normalizer=(
            action_normalizer
        ),
    )

    first_loss = None
    final_loss = None

    for step in range(
        1,
        steps + 1,
    ):
        predicted_actions = policy(
            normalized_obs
        )

        loss, _ = (
            compute_masked_l1_loss(
                predicted_actions=(
                    predicted_actions
                ),
                target_actions=(
                    target_actions
                ),
                is_pad=is_pad,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_grad_norm,
            )

        optimizer.step()

        current_loss = float(
            loss.detach()
        )

        if first_loss is None:
            first_loss = current_loss

        final_loss = current_loss

        if (
            step == 1
            or step % 50 == 0
            or step == steps
        ):
            print(
                f"Overfit step "
                f"{step:04d}/{steps} | "
                f"loss={current_loss:.6f}"
            )

    print("=" * 60)
    print(
        "Initial loss:",
        first_loss,
    )
    print(
        "Final loss:",
        final_loss,
    )
    print("=" * 60)


def save_checkpoint(
    output_path,
    policy,
    observation_normalizer,
    action_normalizer,
    observation_statistics,
    action_statistics,
    obs_shapes,
    action_dim,
    optimizer,
    args,
    epoch,
    validation_loss,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_type": (
            "deterministic_act"
        ),

        "policy_state_dict": (
            policy.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "observation_normalizer_state_dict": (
            observation_normalizer
            .state_dict()
        ),

        "action_normalizer_state_dict": (
            action_normalizer
            .state_dict()
        ),

        "observation_statistics": (
            observation_statistics
        ),

        "action_statistics": (
            action_statistics
        ),

        "low_dim_keys": list(
            LOW_DIM_KEYS
        ),

        "rgb_keys": list(
            RGB_KEYS
        ),

        "obs_shapes": obs_shapes,
        "action_dim": int(
            action_dim
        ),

        "chunk_size": int(
            args.chunk_size
        ),

        "d_model": int(
            args.d_model
        ),

        "nhead": int(
            args.nhead
        ),

        "num_decoder_layers": int(
            args.num_decoder_layers
        ),

        "dim_feedforward": int(
            args.dim_feedforward
        ),

        "dropout": float(
            args.dropout
        ),

        "pretrained_backbone": bool(
            not args.no_pretrained_backbone
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


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Device:", device)
    print("Dataset:", args.dataset)
    print("Chunk size:", args.chunk_size)
    print("d_model:", args.d_model)
    print("Batch size:", args.batch_size)
    print(
        "Pretrained backbone:",
        not args.no_pretrained_backbone,
    )
    print("=" * 60)

    train_dataset = None
    valid_dataset = None

    try:
        train_dataset = (
            ACTRobomimicDataset(
                dataset_path=args.dataset,
                split="train",
                low_dim_keys=(
                    LOW_DIM_KEYS
                ),
                rgb_keys=RGB_KEYS,
                chunk_size=(
                    args.chunk_size
                ),
            )
        )

        valid_dataset = (
            ACTRobomimicDataset(
                dataset_path=args.dataset,
                split="valid",
                low_dim_keys=(
                    LOW_DIM_KEYS
                ),
                rgb_keys=RGB_KEYS,
                chunk_size=(
                    args.chunk_size
                ),
            )
        )

        obs_shapes = (
            train_dataset
            .get_obs_shapes()
        )

        action_dim = int(
            train_dataset
            .get_action_dim()
        )

        observation_statistics = (
            compute_obs_statistics(
                dataset_path=args.dataset,
                split="train",
                obs_keys=LOW_DIM_KEYS,
            )
        )

        action_statistics = (
            compute_action_statistics(
                dataset_path=args.dataset,
                split="train",
            )
        )

        observation_normalizer = (
            ObservationNormalizer(
                statistics=(
                    observation_statistics
                ),
                low_dim_keys=(
                    LOW_DIM_KEYS
                ),
            ).to(device)
        )

        action_normalizer = (
            ActionNormalizer(
                statistics=(
                    action_statistics
                ),
            ).to(device)
        )

        policy = ACTPolicy(
            low_dim_keys=LOW_DIM_KEYS,
            camera_keys=RGB_KEYS,
            obs_shapes=obs_shapes,
            action_dim=action_dim,
            chunk_size=args.chunk_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_decoder_layers=(
                args.num_decoder_layers
            ),
            dim_feedforward=(
                args.dim_feedforward
            ),
            dropout=args.dropout,
            pretrained_backbone=(
                not args.no_pretrained_backbone
            ),
        ).to(device)

        optimizer = build_optimizer(
            policy=policy,
            learning_rate=(
                args.learning_rate
            ),
            backbone_learning_rate=(
                args.backbone_learning_rate
            ),
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
            "Action dimension:",
            action_dim,
        )

        print(
            "Training samples:",
            len(train_dataset),
        )

        print(
            "Validation samples:",
            len(valid_dataset),
        )

        example_batch = next(
            iter(train_loader)
        )

        example_batch_device = (
            batch_to_device(
                example_batch,
                device,
            )
        )

        example_obs = (
            observation_normalizer(
                example_batch_device[
                    "obs"
                ]
            )
        )

        with torch.no_grad():
            example_prediction = (
                policy(example_obs)
            )

        expected_shape = (
            example_batch[
                "actions"
            ].shape
        )

        print(
            "Target actions:",
            tuple(expected_shape),
        )

        print(
            "Prediction:",
            tuple(
                example_prediction.shape
            ),
        )

        if (
            tuple(
                example_prediction.shape
            )
            != tuple(expected_shape)
        ):
            raise RuntimeError(
                "ACT output shape does "
                "not match target shape."
            )

        if args.overfit_one_batch:
            overfit_one_batch(
                policy=policy,
                observation_normalizer=(
                    observation_normalizer
                ),
                action_normalizer=(
                    action_normalizer
                ),
                raw_batch=example_batch,
                optimizer=optimizer,
                device=device,
                max_grad_norm=(
                    args.max_grad_norm
                ),
                steps=(
                    args.overfit_steps
                ),
            )

            return

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
                train_one_epoch(
                    policy=policy,
                    observation_normalizer=(
                        observation_normalizer
                    ),
                    action_normalizer=(
                        action_normalizer
                    ),
                    data_loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    max_grad_norm=(
                        args.max_grad_norm
                    ),
                )
            )

            valid_metrics = validate(
                policy=policy,
                observation_normalizer=(
                    observation_normalizer
                ),
                action_normalizer=(
                    action_normalizer
                ),
                data_loader=valid_loader,
                device=device,
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
                    observation_normalizer=(
                        observation_normalizer
                    ),
                    action_normalizer=(
                        action_normalizer
                    ),
                    observation_statistics=(
                        observation_statistics
                    ),
                    action_statistics=(
                        action_statistics
                    ),
                    obs_shapes=obs_shapes,
                    action_dim=action_dim,
                    optimizer=optimizer,
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
                    "Early stopping after "
                    f"{args.patience} epochs "
                    "without improvement."
                )

                break

        print("=" * 60)
        print(
            "Best validation loss:",
            best_validation_loss,
        )
        print(
            "Checkpoint:",
            args.output,
        )
        print("=" * 60)

    finally:
        if train_dataset is not None:
            train_dataset.close()

        if valid_dataset is not None:
            valid_dataset.close()


if __name__ == "__main__":
    main()