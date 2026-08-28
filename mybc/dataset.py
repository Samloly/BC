import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mybc.observation import OBS_KEYS

def decode_demo_names(values):
    return [
        value.decode("utf-8")
        if isinstance(value, bytes)
        else str(value)
        for value in values
    ]

class RobomimicDataset(Dataset):
    def __init__(self,dataset_path,split,obs_keys):
        self.dataset_path = dataset_path
        self.obs_key = tuple(obs_keys)
        self.split = split
        self.index = []
        self._file = None

        with h5py.File(self.dataset_path,"r") as file:
            demo_names = [
                name.decode() if isinstance(name,bytes) else str(name)
                for name in file["mask"][split][:]
            ]

            for demo_name in demo_names:
                trajectory_length = file["data"][demo_name]["actions"].shape[0]

                for timestep in range(trajectory_length):
                    self.index.append((demo_name,timestep))

        self._file = None

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.dataset_path,"r")

        return self._file
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self,index):
        demo_name, timestep = self.index[index]
        demo = self._get_file()["data"][demo_name]

        observation = {
            key:torch.as_tensor(
                demo["obs"][key][timestep],
                dtype=torch.float32,
            )
            for key in self.obs_key
        }

        action = torch.as_tensor(
            demo["actions"][timestep],
            dtype=torch.float32,
        )

        return {
            "obs":observation,
            "action":action,
            "demo_name":demo_name,
            "timestep":timestep,
        }
    
    def get_obs_shape(self):
        demo_name,timestep = self.index[0]

        with h5py.File(self.dataset_path,"r") as file:
            demo_obs = file["data"][demo_name]["obs"]
            return {
                key: int(
                    demo_obs[key][timestep].size
                )
                for key in self.obs_key
            }
        
    def get_action_dim(self):
        demo_name,timestep = self.index[0]

        with h5py.File(self.dataset_path,"r") as file:
            action = file["data"][demo_name]["actions"][timestep]
            return int(action.size)
    

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        return state
    
    def __del__(self):
        self.close()
        # super().__init__()