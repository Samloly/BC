import argparse
import time
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from mybc.normalizer import (
    ObservationNormalizer,
)
from mybc.policy import (
    TransformerPolicy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained causal "
            "BC-Transformer policy."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/"
            "best_bc_transformer.pth"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Robomimic dataset used to "
            "recover environment metadata."
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
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--render-delay",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--camera-name",
        type=str,
        default="agentview",
    )

    parser.add_argument(
        "--video-fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    return parser.parse_args()


def validate_args(args):
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{args.checkpoint}"
        )

    if not args.dataset.exists():
        raise FileNotFoundError(
            "Dataset not found: "
            f"{args.dataset}"
        )

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be positive"
        )

    if args.horizon <= 0:
        raise ValueError(
            "--horizon must be positive"
        )

    if args.render_delay < 0:
        raise ValueError(
            "--render-delay must be "
            "non-negative"
        )

    if args.video_fps <= 0:
        raise ValueError(
            "--video-fps must be positive"
        )


def set_seed(seed):
    if seed is None:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_observation_history(
    obs_keys,
    sequence_length,
):
    """
    Create one fixed-length history buffer
    for every observation key.
    """
    return {
        key: deque(
            maxlen=sequence_length
        )
        for key in obs_keys
    }


def append_observation(
    history,
    observation,
    obs_keys,
):
    """
    Append the current environment observation
    to the history.

    A copy is stored because some environments
    may reuse their observation arrays.
    """
    for key in obs_keys:
        if key not in observation:
            raise KeyError(
                "Environment observation does "
                f"not contain key: {key}"
            )

        value = np.asarray(
            observation[key]
        ).copy()

        history[key].append(value)


def history_to_tensor(
    history,
    obs_keys,
    device,
):
    """
    Convert history to tensors.

    Output for each key:
        [1, T, *obs_shape]
    """
    tensor_observation = {}

    for key in obs_keys:
        if len(history[key]) == 0:
            raise RuntimeError(
                "Cannot convert an empty "
                "observation history."
            )

        sequence = np.stack(
            list(history[key]),
            axis=0,
        )

        tensor_observation[key] = (
            torch.as_tensor(
                sequence,
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
        )

    return tensor_observation


def close_environment(env):
    if hasattr(env, "close"):
        env.close()

    elif (
        hasattr(env, "env")
        and hasattr(env.env, "close")
    ):
        env.env.close()


def load_policy_and_normalizer(
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

    if model_type != "bc_transformer":
        raise ValueError(
            "Expected a bc_transformer "
            "checkpoint, but received: "
            f"{model_type}"
        )

    required_keys = (
        "policy_state_dict",
        "normalizer_state_dict",
        "observation_statistics",
        "obs_keys",
        "obs_shapes",
        "action_dim",
        "sequence_length",
        "d_model",
        "nhead",
        "num_layers",
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

    obs_keys = tuple(
        checkpoint["obs_keys"]
    )

    sequence_length = int(
        checkpoint["sequence_length"]
    )

    policy = TransformerPolicy(
        obs_shapes=checkpoint[
            "obs_shapes"
        ],
        obs_keys=obs_keys,
        action_dim=int(
            checkpoint["action_dim"]
        ),
        sequence_length=sequence_length,
        d_model=int(
            checkpoint["d_model"]
        ),
        nhead=int(
            checkpoint["nhead"]
        ),
        num_layers=int(
            checkpoint["num_layers"]
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
    ).to(device)

    policy.load_state_dict(
        checkpoint["policy_state_dict"]
    )

    policy.eval()

    normalizer = ObservationNormalizer(
        statistics=checkpoint[
            "observation_statistics"
        ],
        obs_keys=obs_keys,
    ).to(device)

    normalizer.load_state_dict(
        checkpoint[
            "normalizer_state_dict"
        ]
    )

    normalizer.eval()

    return (
        policy,
        normalizer,
        obs_keys,
        sequence_length,
        checkpoint,
    )


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
        normalizer,
        obs_keys,
        sequence_length,
        checkpoint,
    ) = load_policy_and_normalizer(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    print("=" * 60)
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Dataset:", args.dataset)
    print("Observation keys:", obs_keys)
    print(
        "Sequence length:",
        sequence_length,
    )
    print(
        "Action dimension:",
        checkpoint["action_dim"],
    )
    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )
    print(
        "Checkpoint validation loss:",
        checkpoint.get(
            "validation_loss",
            "unknown",
        ),
    )
    print("=" * 60)

    ObsUtils.initialize_obs_modality_mapping_from_dict(
        modality_mapping={
            "low_dim": list(obs_keys),
            "rgb": [],
            "depth": [],
            "scan": [],
        }
    )

    env_metadata = (
        FileUtils
        .get_env_metadata_from_dataset(
            dataset_path=str(
                args.dataset
            )
        )
    )

    save_video = (
        args.video_dir is not None
    )

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_metadata,
        render=args.render,
        render_offscreen=save_video,
        use_image_obs=False,
    )

    if save_video:
        args.video_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    success_count = 0
    total_return = 0.0
    total_steps = 0

    try:
        for episode in range(
            args.episodes
        ):
            observation = env.reset()

            observation_history = (
                create_observation_history(
                    obs_keys=obs_keys,
                    sequence_length=(
                        sequence_length
                    ),
                )
            )

            episode_success = False
            episode_return = 0.0
            episode_steps = 0
            video_writer = None

            if save_video:
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
                for timestep in range(
                    args.horizon
                ):
                    append_observation(
                        history=(
                            observation_history
                        ),
                        observation=observation,
                        obs_keys=obs_keys,
                    )

                    tensor_observation = (
                        history_to_tensor(
                            history=(
                                observation_history
                            ),
                            obs_keys=obs_keys,
                            device=device,
                        )
                    )

                    with torch.inference_mode():
                        normalized_observation = (
                            normalizer(
                                tensor_observation
                            )
                        )

                        predicted_actions = (
                            policy(
                                normalized_observation,
                                padding_mask=None,
                            )
                        )

                    # The last token corresponds
                    # to the current timestep.
                    action = (
                        predicted_actions[
                            0, -1
                        ]
                        .cpu()
                        .numpy()
                    )

                    action = np.clip(
                        action,
                        -1.0,
                        1.0,
                    )

                    (
                        observation,
                        reward,
                        done,
                        info,
                    ) = env.step(action)

                    episode_return += float(
                        reward
                    )

                    episode_steps = (
                        timestep + 1
                    )

                    if args.render:
                        env.render(
                            mode="human"
                        )

                        if (
                            args.render_delay
                            > 0
                        ):
                            time.sleep(
                                args.render_delay
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
                                args.camera_name
                            ),
                        )

                        video_writer.append_data(
                            frame
                        )

                    success_result = (
                        env.is_success()
                    )

                    episode_success = bool(
                        success_result["task"]
                    )

                    if (
                        episode_success
                        or done
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

            total_steps += episode_steps

            print(
                f"Episode "
                f"{episode + 1:03d} | "
                f"{'SUCCESS' if episode_success else 'FAIL'} | "
                f"return={episode_return:.3f} | "
                f"steps={episode_steps}"
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