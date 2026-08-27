import numpy as np
import torch

OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]

def concatenate_obs(obs):
    return np.concatenate(
        [np.asarray(obs[key],dtype=np.float32) for key in OBS_KEYS],
        axis=-1,
    )

def concatenate_tensor_obs(obs):
    return torch.cat(
        [obs[key].float() for key in OBS_KEYS],
        dim=-1,
    )