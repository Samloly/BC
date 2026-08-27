import argparse
import random
from pathlib import Path

import numpy as np
import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from mybc.normalizer import (
    ObservationNormalizer,
)
from mybc.policy import MLPPolicy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/best_bc_mlp.pth"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def observation_to_tensor(
    observation,
    obs_keys,
    device,
):
    tensor_observation = {}

    for key in obs_keys:
        if key not in observation:
            available_keys = list(
                observation.keys()
            )

            raise KeyError(
                f"Environment observation does not "
                f"contain {key}. Available keys: "
                f"{available_keys}"
            )

        tensor_observation[key] = (
            torch.as_tensor(
                observation[key],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
        )

    return tensor_observation


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    dataset_path = args.dataset

    if dataset_path is None:
        dataset_path = Path(
            checkpoint["dataset_path"]
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    obs_keys = tuple(
        checkpoint["obs_keys"]
    )

    policy = MLPPolicy(
        obs_shapes=checkpoint["obs_shapes"],
        obs_keys=obs_keys,
        action_dim=checkpoint["action_dim"],
        hidden_dims=checkpoint[
            "hidden_dims"
        ],
    ).to(device)

    policy.load_state_dict(
        checkpoint["policy_state_dict"]
    )

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

    policy.eval()
    normalizer.eval()

    env_metadata = (
        FileUtils.get_env_metadata_from_dataset(
            dataset_path=str(dataset_path)
        )
    )

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_metadata,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )

    success_count = 0
    episode_lengths = []
    maximum_cube_heights = []

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Dataset:", dataset_path)
    print("Observation keys:", obs_keys)
    print(
        f"Evaluating {args.episodes} episodes"
    )

    for episode in range(args.episodes):
        observation = env.reset()

        episode_success = False
        episode_length = 0
        maximum_cube_height = float(
            "-inf"
        )

        for timestep in range(args.horizon):
            tensor_observation = (
                observation_to_tensor(
                    observation=observation,
                    obs_keys=obs_keys,
                    device=device,
                )
            )

            with torch.no_grad():
                normalized_observation = (
                    normalizer(
                        tensor_observation
                    )
                )

                action = policy(
                    normalized_observation
                )

            action = (
                action.squeeze(0)
                .cpu()
                .numpy()
            )

            action = np.clip(
                action,
                -1.0,
                1.0,
            )

            observation, reward, done, info = (
                env.step(action)
            )

            episode_length = timestep + 1

            if "object" in observation:
                # In Lift, object[:3] commonly contains
                # object position-related values, but the
                # exact semantic layout depends on dataset.
                # Task success below remains authoritative.
                object_values = np.asarray(
                    observation["object"]
                )

                if object_values.size >= 3:
                    maximum_cube_height = max(
                        maximum_cube_height,
                        float(object_values[2]),
                    )

            success = env.is_success()
            episode_success = bool(
                success.get("task", False)
            )

            if episode_success or done:
                break

        if episode_success:
            success_count += 1

        episode_lengths.append(
            episode_length
        )

        maximum_cube_heights.append(
            maximum_cube_height
        )

        status = (
            "SUCCESS"
            if episode_success
            else "FAIL"
        )

        print(
            f"Episode {episode + 1:03d} | "
            f"{status} | "
            f"steps={episode_length}"
        )

    success_rate = (
        success_count
        / args.episodes
    )

    print("=" * 50)
    print(
        f"Success: {success_count}/"
        f"{args.episodes}"
    )
    print(
        f"Success rate: "
        f"{100.0 * success_rate:.2f}%"
    )
    print(
        f"Average episode length: "
        f"{np.mean(episode_lengths):.2f}"
    )
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()