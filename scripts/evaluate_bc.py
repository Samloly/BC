import argparse
import random
from pathlib import Path

import numpy as np
import torch

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from mybc.normalizer import ObservationNormalizer
from mybc.policy import MLPPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a BC MLP policy in robosuite."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_bc_mlp.pth"),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset used to recover robosuite environment metadata.",
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
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="If provided, save one MP4 file for each episode.",
    )

    parser.add_argument(
        "--video-height",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--video-width",
        type=int,
        default=512,
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
        "--debug",
        action="store_true",
        help="Print observation and action information on the first step.",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_argument):
    if device_argument == "cpu":
        return torch.device("cpu")

    if device_argument == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but CUDA is unavailable."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def initialize_observation_utils(obs_keys):
    """
    Register every observation used by this low-dimensional BC policy.

    EnvRobosuite.get_observation() queries this global mapping, so it
    must be initialized before env.reset().
    """
    modality_mapping = {
        "low_dim": list(obs_keys),
        "rgb": [],
        "depth": [],
        "scan": [],
    }

    ObsUtils.initialize_obs_modality_mapping_from_dict(
        modality_mapping=modality_mapping
    )

    if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
        raise RuntimeError(
            "Failed to initialize robomimic observation modalities."
        )

    missing_keys = [
        key
        for key in obs_keys
        if key not in ObsUtils.OBS_KEYS_TO_MODALITIES
    ]

    if missing_keys:
        raise RuntimeError(
            f"Observation modality mapping is missing: {missing_keys}"
        )


def observation_to_tensor(
    observation,
    obs_keys,
    device,
):
    tensor_observation = {}

    for key in obs_keys:
        if key not in observation:
            raise KeyError(
                f"Environment observation does not contain '{key}'. "
                f"Available keys: {sorted(observation.keys())}"
            )

        value = torch.as_tensor(
            observation[key],
            dtype=torch.float32,
            device=device,
        )

        # Add batch dimension:
        # (3,) -> (1, 3)
        value = value.unsqueeze(0)

        if not torch.isfinite(value).all():
            raise ValueError(
                f"Observation '{key}' contains NaN or Inf."
            )

        tensor_observation[key] = value

    return tensor_observation


def load_policy_and_normalizer(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = (
        "policy_state_dict",
        "normalizer_state_dict",
        "observation_statistics",
        "obs_keys",
        "obs_shapes",
        "action_dim",
        "hidden_dims",
    )

    missing_keys = [
        key
        for key in required_keys
        if key not in checkpoint
    ]

    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing required fields: {missing_keys}. "
            f"Available fields: {list(checkpoint.keys())}"
        )

    obs_keys = tuple(checkpoint["obs_keys"])

    policy = MLPPolicy(
        obs_shapes=checkpoint["obs_shapes"],
        obs_keys=obs_keys,
        action_dim=int(checkpoint["action_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
    ).to(device)

    policy.load_state_dict(
        checkpoint["policy_state_dict"]
    )

    normalizer = ObservationNormalizer(
        statistics=checkpoint["observation_statistics"],
        obs_keys=obs_keys,
    ).to(device)

    normalizer.load_state_dict(
        checkpoint["normalizer_state_dict"]
    )

    policy.eval()
    normalizer.eval()

    return checkpoint, policy, normalizer, obs_keys


def render_frame(
    env,
    camera_name,
    height,
    width,
):
    frame = env.render(
        mode="rgb_array",
        camera_name=camera_name,
        height=height,
        width=width,
    )

    frame = np.asarray(frame)

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def run_episode(
    env,
    policy,
    normalizer,
    obs_keys,
    action_dim,
    device,
    horizon,
    debug=False,
    video_path=None,
    camera_name="agentview",
    video_height=512,
    video_width=512,
    video_fps=20,
):
    video_writer = None

    if video_path is not None:
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise ImportError(
                "Video recording requires imageio. "
                "Install it with: python -m pip install imageio imageio-ffmpeg"
            ) from error

        video_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        video_writer = imageio.get_writer(
            str(video_path),
            fps=video_fps,
        )

    try:
        observation = env.reset()

        if debug:
            print(
                "Environment observation keys:",
                sorted(observation.keys()),
            )

        episode_return = 0.0
        episode_success = False
        episode_length = 0

        if video_writer is not None:
            video_writer.append_data(
                render_frame(
                    env=env,
                    camera_name=camera_name,
                    height=video_height,
                    width=video_width,
                )
            )

        for timestep in range(horizon):
            tensor_observation = observation_to_tensor(
                observation=observation,
                obs_keys=obs_keys,
                device=device,
            )

            with torch.no_grad():
                normalized_observation = normalizer(
                    tensor_observation
                )

                predicted_action = policy(
                    normalized_observation
                )

            if predicted_action.shape != (1, action_dim):
                raise RuntimeError(
                    "Unexpected policy output shape: "
                    f"{tuple(predicted_action.shape)}; "
                    f"expected: {(1, action_dim)}"
                )

            if not torch.isfinite(predicted_action).all():
                raise ValueError(
                    "Policy produced an action containing NaN or Inf."
                )

            action = (
                predicted_action[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # The training actions are already in robosuite's
            # normalized control range.
            action = np.clip(action, -1.0, 1.0)

            if debug and timestep == 0:
                print(
                    "Observation tensor shapes:",
                    {
                        key: tuple(value.shape)
                        for key, value
                        in tensor_observation.items()
                    },
                )
                print("First predicted action:", action)
                print(
                    "Action range:",
                    float(action.min()),
                    float(action.max()),
                )

            observation, reward, done, info = env.step(
                action
            )

            episode_return += float(reward)
            episode_length = timestep + 1

            if video_writer is not None:
                video_writer.append_data(
                    render_frame(
                        env=env,
                        camera_name=camera_name,
                        height=video_height,
                        width=video_width,
                    )
                )

            success_information = env.is_success()

            episode_success = bool(
                success_information.get("task", False)
            )

            if episode_success or done:
                break

        return {
            "success": episode_success,
            "length": episode_length,
            "return": episode_return,
        }

    finally:
        if video_writer is not None:
            video_writer.close()



def main():
    args = parse_args()
    set_seed(args.seed)

    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")

    if args.horizon <= 0:
        raise ValueError("--horizon must be greater than zero.")

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    device = get_device(args.device)

    checkpoint, policy, normalizer, obs_keys = (
        load_policy_and_normalizer(
            checkpoint_path=args.checkpoint,
            device=device,
        )
    )

    dataset_path = args.dataset

    if dataset_path is None:
        checkpoint_dataset = checkpoint.get(
            "dataset_path"
        )

        if checkpoint_dataset is None:
            raise KeyError(
                "No --dataset was provided and the checkpoint "
                "does not contain dataset_path."
            )

        dataset_path = Path(checkpoint_dataset)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    # This must happen before env.reset().
    initialize_observation_utils(obs_keys)

    env_metadata = (
        FileUtils.get_env_metadata_from_dataset(
            dataset_path=str(dataset_path)
        )
    )

    record_video = args.video_dir is not None

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_metadata,
        render=False,
        render_offscreen=record_video,
        use_image_obs=False,
    )

    action_dim = int(checkpoint["action_dim"])

    print("=" * 60)
    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )
    print(
        "Checkpoint validation loss:",
        checkpoint.get("validation_loss", "unknown"),
    )
    print("Dataset:", dataset_path)
    print("Environment:", env_metadata.get("env_name"))
    print("Observation keys:", obs_keys)
    print("Observation shapes:", checkpoint["obs_shapes"])
    print("Action dimension:", action_dim)
    print(
        "Observation modalities:",
        ObsUtils.OBS_KEYS_TO_MODALITIES,
    )
    print("Episodes:", args.episodes)
    print("Horizon:", args.horizon)
    print("=" * 60)

    episode_results = []

    try:
        for episode_index in range(args.episodes):
            video_path = None

            if args.video_dir is not None:
                video_path = (
                    args.video_dir
                    / f"episode_{episode_index + 1:03d}.mp4"
                )

            result = run_episode(
                env=env,
                policy=policy,
                normalizer=normalizer,
                obs_keys=obs_keys,
                action_dim=action_dim,
                device=device,
                horizon=args.horizon,
                debug=(
                    args.debug
                    and episode_index == 0
                ),
                video_path=video_path,
                camera_name=args.camera_name,
                video_height=args.video_height,
                video_width=args.video_width,
                video_fps=args.video_fps,
            )

            episode_results.append(result)

            status = (
                "SUCCESS"
                if result["success"]
                else "FAIL"
            )

            print(
                f"Episode {episode_index + 1:03d} | "
                f"{status} | "
                f"steps={result['length']} | "
                f"return={result['return']:.3f}"
            )

    finally:
        if hasattr(env, "env") and hasattr(env.env, "close"):
            env.env.close()

    successes = sum(
        int(result["success"])
        for result in episode_results
    )

    success_rate = successes / len(
        episode_results
    )

    average_length = np.mean(
        [
            result["length"]
            for result in episode_results
        ]
    )

    average_return = np.mean(
        [
            result["return"]
            for result in episode_results
        ]
    )

    print("=" * 60)
    print(
        f"Success: {successes}/{len(episode_results)}"
    )
    print(
        f"Success rate: {100.0 * success_rate:.2f}%"
    )
    print(
        f"Average episode length: {average_length:.2f}"
    )
    print(
        f"Average return: {average_return:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()