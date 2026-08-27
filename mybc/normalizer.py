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
            "mean" : mean(axis=0).astype(np.float32),
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
            obs_keys, 
        ):
        super().__init__()

        self.obs_keys = obs_keys

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
        normalized = {}

        for key in self.obs_keys:
            mean = getattr(self, f"{key}_mean")
            std = getattr(self, f"{key}_std")

            normalized[key] = (observation[key]-mean)/std
        return normalized