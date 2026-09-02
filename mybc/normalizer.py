import h5py
import numpy as np
import torch
import torch.nn as nn
from mybc.dataset import decode_demo_names


def compute_obs_statistics(
    
        dataset_path,
        split,
        obs_keys
):
    values_by_key = {
        key :[]
        for key in obs_keys
    }

    with h5py.File(dataset_path,"r") as file:
        demo_names = decode_demo_names(
            file["mask"][split][:]
        )

        for demo_name  in demo_names:
            demo_obs = file["data"][demo_name]["obs"]

            for key in obs_keys:
                values_by_key[key].append(
                    np.asarray(
                        demo_obs[key],
                        dtype=np.float32,
                    )
                )

    statistics = {}

    for key in obs_keys:
        values = np.concatenate(
            values_by_key[key],
            axis=0,
        )

        mean = values.mean(axis=0)
        std = values.std(axis=0)

        statistics[key] = {
            "mean" : mean.astype(np.float32),
            "std": np.maximum(
                std,
                1e-6,
            ).astype(np.float32)
        }
    return statistics

class ObservationNormalizer(nn.Module):
    def __init__(
            self,
            statistics,
            obs_keys=None,
            low_dim_keys=None, 
        ):
        super().__init__()

        selected_keys = low_dim_keys if low_dim_keys is not None else obs_keys
        if selected_keys == None:
            selected_keys = statistics.keys()
        
        self.low_dim_keys = selected_keys


        self.obs_keys = self.low_dim_keys

        for key in self.obs_keys:
            mean = torch.as_tensor(
                statistics[key]["mean"],
                dtype=torch.float32,
            )

            std = torch.as_tensor(
                statistics[key]["std"],
                dtype=torch.float32
            )

            self.register_buffer(
                f"{key}_mean",
                mean,
            )

            self.register_buffer(
                f"{key}_std",
                std,
            )
    
    def forward(self,observation):
        normalized = dict(observation)

        for key in self.obs_keys:
            mean = getattr(self, f"{key}_mean")
            std = getattr(self, f"{key}_std")

            normalized[key] = (observation[key]-mean)/std
        return normalized
    
def compute_action_statistics(dataset_path, split):
    action_values = []
    with h5py.File(dataset_path,"r") as file:
        demo_names = decode_demo_names(file["mask"][split][:])
        for demo_name in demo_names:
            demo = file["data"][demo_name]
            actions = np.asarray(demo["actions"],dtype=np.float32)
            action_values.append(actions)
    
    values = np.concatenate(action_values,axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.maximum(std,1e-6).astype(np.float32)
    return {
        "mean":mean,
        "std":std,
        "count":values.shape[0]
    }
    
class ActionNormalizer(nn.Module):
    def __init__(self, statistics):
        super().__init__()

        mean = torch.as_tensor(statistics["mean"],dtype=torch.float32)
        std = torch.as_tensor(statistics["std"],dtype=torch.float32)

        self.register_buffer("action_mean",mean)
        self.register_buffer("action_std",std)
    
    @property
    def action_dim(self):
        return self.action_mean.numel()
    
    def normalize(self,actions):
        return (actions-self.action_mean)/self.action_std
    
    def denormalize(self,actions):
        return actions*self.action_std+self.action_mean
    
    def forward(self,actions):
        return self.normalize(actions)