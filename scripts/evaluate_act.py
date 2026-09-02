import argparse
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from mybc.act import ACTPolicy
from mybc.normalizer import (
    ActionNormalizer,
    ObservationNormalizer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic ACT in "
            "a robomimic environment."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Dataset used to reconstruct the "
            "environment. If omitted, use the "
            "path saved in the checkpoint."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=5,
        help=(
            "Number of predicted actions to "
            "execute before querying ACT again."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--video-camera",
        type=str,
        default="agentview",
    )

    parser.add_argument(
        "--video-fps",
        type=int,
        default=20,
    )

    return parser.parse_args()


def validate_args(args):
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be positive."
        )

    if args.horizon <= 0:
        raise ValueError(
            "--horizon must be positive."
        )

    if args.execution_horizon <= 0:
        raise ValueError(
            "--execution-horizon must be "
            "positive."
        )

    if args.video_fps <= 0:
        raise ValueError(
            "--video-fps must be positive."
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def rgb_to_tensor(
    image,
    device,
):
    image = torch.as_tensor(
        image,
        dtype=torch.float32,
        device=device,
    )

    if image.ndim != 3:
        raise ValueError(
            "Expected RGB image with "
            f"3 dimensions, got "
            f"{tuple(image.shape)}."
        )

    # HWC -> CHW
    if image.shape[-1] in (3, 4):
        image = image.permute(
            2, 0, 1
        )

    if image.shape[0] == 4:
        image = image[:3]

    if image.shape[0] != 3:
        raise ValueError(
            "Expected RGB image with shape "
            f"[3,H,W], got "
            f"{tuple(image.shape)}."
        )

    if image.max() > 1.0:
        image = image / 255.0

    # [3,H,W] -> [1,3,H,W]
    return (
        image.contiguous()
        .unsqueeze(0)
    )


def observation_to_tensor(
    observation,
    low_dim_keys,
    rgb_keys,
    device,
):
    tensor_observation = {}

    required_keys = (
        tuple(low_dim_keys)
        + tuple(rgb_keys)
    )

    missing_keys = [
        key
        for key in required_keys
        if key not in observation
    ]

    if missing_keys:
        raise KeyError(
            "Environment observation is "
            f"missing keys {missing_keys}. "
            "Available keys: "
            f"{list(observation.keys())}"
        )

    for key in low_dim_keys:
        tensor_observation[key] = (
            torch.as_tensor(
                observation[key],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
        )

    for key in rgb_keys:
        tensor_observation[key] = (
            rgb_to_tensor(
                observation[key],
                device=device,
            )
        )

    return tensor_observation


def load_policy(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model_type = checkpoint.get(
        "model_type"
    )

    if model_type not in (
        "deterministic_act",
        "act",
    ):
        raise ValueError(
            "Expected a deterministic ACT "
            f"checkpoint, got {model_type!r}."
        )

    required_keys = (
        "policy_state_dict",
        "observation_statistics",
        "action_statistics",
        "low_dim_keys",
        "rgb_keys",
        "obs_shapes",
        "action_dim",
        "chunk_size",
        "d_model",
        "nhead",
        "num_decoder_layers",
        "dim_feedforward",
    )

    missing_keys = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing keys: "
            f"{missing_keys}"
        )

    low_dim_keys = tuple(
        checkpoint["low_dim_keys"]
    )

    rgb_keys = tuple(
        checkpoint["rgb_keys"]
    )

    # 不在加载checkpoint前再次下载预训练权重。
    # Backbone权重已包含在policy_state_dict中。
    policy = ACTPolicy(
        low_dim_keys=low_dim_keys,
        camera_keys=rgb_keys,
        obs_shapes=checkpoint[
            "obs_shapes"
        ],
        action_dim=int(
            checkpoint["action_dim"]
        ),
        chunk_size=int(
            checkpoint["chunk_size"]
        ),
        d_model=int(
            checkpoint["d_model"]
        ),
        nhead=int(
            checkpoint["nhead"]
        ),
        num_decoder_layers=int(
            checkpoint[
                "num_decoder_layers"
            ]
        ),
        dim_feedforward=int(
            checkpoint[
                "dim_feedforward"
            ]
        ),
        dropout=float(
            checkpoint.get(
                "dropout",
                0.1,
            )
        ),
        pretrained_backbone=False,
    ).to(device)

    policy.load_state_dict(
        checkpoint[
            "policy_state_dict"
        ]
    )

    observation_normalizer = (
        ObservationNormalizer(
            statistics=checkpoint[
                "observation_statistics"
            ],
            low_dim_keys=low_dim_keys,
        ).to(device)
    )

    if (
        "observation_normalizer_state_dict"
        in checkpoint
    ):
        observation_normalizer.load_state_dict(
            checkpoint[
                "observation_normalizer_state_dict"
            ]
        )

    action_normalizer = (
        ActionNormalizer(
            statistics=checkpoint[
                "action_statistics"
            ]
        ).to(device)
    )

    if (
        "action_normalizer_state_dict"
        in checkpoint
    ):
        action_normalizer.load_state_dict(
            checkpoint[
                "action_normalizer_state_dict"
            ]
        )

    policy.eval()
    observation_normalizer.eval()
    action_normalizer.eval()

    return (
        policy,
        observation_normalizer,
        action_normalizer,
        low_dim_keys,
        rgb_keys,
        checkpoint,
    )


def create_environment(
    dataset_path,
    low_dim_keys,
    rgb_keys,
    render,
):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        modality_mapping={
            "low_dim": list(
                low_dim_keys
            ),
            "rgb": list(
                rgb_keys
            ),
            "depth": [],
            "scan": [],
        }
    )

    env_metadata = (
        FileUtils
        .get_env_metadata_from_dataset(
            dataset_path=str(
                dataset_path
            )
        )
    )

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_metadata,
        render=render,
        render_offscreen=True,
        use_image_obs=True,
    )

    return env


def close_environment(env):
    if hasattr(env, "close"):
        env.close()
    elif (
        hasattr(env, "env")
        and hasattr(env.env, "close")
    ):
        env.env.close()


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        policy,
        observation_normalizer,
        action_normalizer,
        low_dim_keys,
        rgb_keys,
        checkpoint,
    ) = load_policy(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    dataset_path = args.dataset

    if dataset_path is None:
        saved_dataset_path = (
            checkpoint.get(
                "dataset_path"
            )
        )

        if saved_dataset_path is None:
            raise ValueError(
                "No --dataset was provided "
                "and the checkpoint does not "
                "contain dataset_path."
            )

        dataset_path = Path(
            saved_dataset_path
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{dataset_path}"
        )

    chunk_size = int(
        checkpoint["chunk_size"]
    )

    if (
        args.execution_horizon
        > chunk_size
    ):
        raise ValueError(
            "--execution-horizon cannot "
            f"exceed chunk_size={chunk_size}."
        )

    env = create_environment(
        dataset_path=dataset_path,
        low_dim_keys=low_dim_keys,
        rgb_keys=rgb_keys,
        render=args.render,
    )

    if args.video_dir is not None:
        args.video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("=" * 60)
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Dataset:", dataset_path)
    print("Low-dimensional keys:", low_dim_keys)
    print("RGB keys:", rgb_keys)
    print("Chunk size:", chunk_size)
    print(
        "Execution horizon:",
        args.execution_horizon,
    )
    print("Episodes:", args.episodes)
    print("Horizon:", args.horizon)
    print("=" * 60)

    success_count = 0
    total_return = 0.0
    total_steps = 0

    try:
        for episode in range(
            args.episodes
        ):
            observation = env.reset()

            episode_success = False
            episode_return = 0.0
            episode_steps = 0
            number_of_queries = 0

            video_writer = None

            if args.video_dir is not None:
                video_path = (
                    args.video_dir
                    / (
                        f"episode_"
                        f"{episode + 1:03d}.mp4"
                    )
                )

                video_writer = (
                    imageio.get_writer(
                        str(video_path),
                        fps=args.video_fps,
                    )
                )

            try:
                while (
                    episode_steps
                    < args.horizon
                    and not episode_success
                ):
                    tensor_observation = (
                        observation_to_tensor(
                            observation=(
                                observation
                            ),
                            low_dim_keys=(
                                low_dim_keys
                            ),
                            rgb_keys=rgb_keys,
                            device=device,
                        )
                    )

                    normalized_observation = (
                        observation_normalizer(
                            tensor_observation
                        )
                    )

                    with torch.inference_mode():
                        normalized_action_chunk = (
                            policy(
                                normalized_observation
                            )
                        )

                        action_chunk = (
                            action_normalizer
                            .denormalize(
                                normalized_action_chunk
                            )
                        )

                        action_chunk = (
                            torch.clamp(
                                action_chunk,
                                -1.0,
                                1.0,
                            )
                        )

                    # [1,K,A] -> [K,A]
                    action_chunk = (
                        action_chunk[0]
                        .cpu()
                        .numpy()
                    )

                    number_of_queries += 1

                    actions_to_execute = min(
                        args.execution_horizon,
                        action_chunk.shape[0],
                        (
                            args.horizon
                            - episode_steps
                        ),
                    )

                    for action_index in range(
                        actions_to_execute
                    ):
                        action = action_chunk[
                            action_index
                        ]

                        (
                            observation,
                            reward,
                            done,
                            info,
                        ) = env.step(action)

                        episode_return += float(
                            reward
                        )

                        episode_steps += 1

                        if args.render:
                            env.render(
                                mode="human"
                            )

                        if (
                            video_writer
                            is not None
                        ):
                            frame = env.render(
                                mode="rgb_array",
                                height=512,
                                width=512,
                                camera_name=(
                                    args.video_camera
                                ),
                            )

                            video_writer.append_data(
                                frame
                            )

                        success_result = (
                            env.is_success()
                        )

                        episode_success = bool(
                            success_result.get(
                                "task",
                                False,
                            )
                        )

                        if (
                            episode_success
                            or done
                            or episode_steps
                            >= args.horizon
                        ):
                            break

            finally:
                if video_writer is not None:
                    video_writer.close()

            success_count += int(
                episode_success
            )

            total_return += (
                episode_return
            )

            total_steps += (
                episode_steps
            )

            status = (
                "SUCCESS"
                if episode_success
                else "FAIL"
            )

            print(
                f"Episode "
                f"{episode + 1:03d} | "
                f"{status} | "
                f"return="
                f"{episode_return:.3f} | "
                f"steps={episode_steps} | "
                f"queries={number_of_queries}"
            )

    finally:
        close_environment(env)

    success_rate = (
        success_count
        / args.episodes
    )

    average_return = (
        total_return
        / args.episodes
    )

    average_steps = (
        total_steps
        / args.episodes
    )

    print("=" * 60)
    print(
        f"Success: "
        f"{success_count}/"
        f"{args.episodes}"
    )
    print(
        f"Success rate: "
        f"{100.0 * success_rate:.2f}%"
    )
    print(
        f"Average return: "
        f"{average_return:.3f}"
    )
    print(
        f"Average steps: "
        f"{average_steps:.2f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()