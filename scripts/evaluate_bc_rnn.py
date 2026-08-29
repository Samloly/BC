import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from mybc.normalizer import ObservationNormalizer
from mybc.policy import RNNPolicy


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/best_bc_rnn.pth"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
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

    return parser.parse_args()


def observation_to_tensor(
    observation,
    obs_keys,
    device,
):
    return {
        key: torch.as_tensor(
            observation[key],
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
        for key in obs_keys
    }


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

    obs_keys = tuple(
        checkpoint["obs_keys"]
    )

    policy = RNNPolicy(
        obs_shapes=checkpoint[
            "obs_shapes"
        ],
        obs_keys=obs_keys,
        action_dim=checkpoint[
            "action_dim"
        ],
        hidden_size=checkpoint[
            "hidden_size"
        ],
        num_layers=checkpoint[
            "num_layers"
        ],
        dropout=checkpoint.get(
            "dropout",
            0.0,
        ),
    ).to(device)

    policy.load_state_dict(
        checkpoint[
            "policy_state_dict"
        ]
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

    try:
        for episode in range(
            args.episodes
        ):
            observation = env.reset()
            hidden_state = None
            episode_success = False
            video_writer = None

            if save_video:
                video_path = (
                    args.video_dir
                    / f"episode_{episode + 1:03d}.mp4"
                )

                video_writer = (
                    imageio.get_writer(
                        str(video_path),
                        fps=20,
                    )
                )

            try:
                for timestep in range(
                    args.horizon
                ):
                    tensor_observation = (
                        observation_to_tensor(
                            observation,
                            obs_keys,
                            device,
                        )
                    )

                    with torch.no_grad():
                        normalized_observation = (
                            normalizer(
                                tensor_observation
                            )
                        )

                        (
                            predicted_actions,
                            hidden_state,
                        ) = policy(
                            normalized_observation,
                            hidden_state,
                        )

                    action = (
                        predicted_actions[0, 0]
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

                    if args.render:
                        env.render(
                            mode="human"
                        )

                        time.sleep(
                            args.render_delay
                        )

                    if video_writer is not None:
                        frame = env.render(
                            mode="rgb_array",
                            height=512,
                            width=512,
                            camera_name="agentview",
                        )

                        video_writer.append_data(
                            frame
                        )

                    episode_success = (
                        env.is_success()[
                            "task"
                        ]
                    )

                    if episode_success or done:
                        break

            finally:
                if video_writer is not None:
                    video_writer.close()

            success_count += int(
                episode_success
            )

            print(
                f"Episode "
                f"{episode + 1:03d} | "
                f"{'SUCCESS' if episode_success else 'FAIL'} | "
                f"steps={timestep + 1}"
            )

    finally:
        close_environment(env)

    print(
        f"Success: "
        f"{success_count}/"
        f"{args.episodes}"
    )

    print(
        f"Success rate: "
        f"{100.0 * success_count / args.episodes:.2f}%"
    )


if __name__ == "__main__":
    main()