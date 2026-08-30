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

        actions = torch.as_tensor(
            demo["actions"][timestep],
            dtype=torch.float32,
        )

        return {
            "obs":observation,
            "actions":actions,
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
            actions = file["data"][demo_name]["actions"][timestep]
            return int(actions.size)
    

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state
    
    def __del__(self):
        self.close()
        # super().__init__()

class SequenceRobomimicDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        split,
        obs_keys,
        sequence_length=10,
        pad_sequence=True,
        get_pad_mask = False,
        ):
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.obs_keys = obs_keys
        self.sequence_length = sequence_length
        self.pad_sequence = pad_sequence
        self.get_pad_mask = get_pad_mask
        self.index = []
        self._file =None
        
        with h5py.File(self.dataset_path, "r") as file:
            demo_names = decode_demo_names(file["mask"][self.split][:])

            for demo_name in demo_names:
                demo = file["data"][demo_name]
                trajectory_length = demo["actions"].shape[0]
                if self.pad_sequence:
                    number_of_sequence = trajectory_length
                else:
                    number_of_sequence = max(trajectory_length-self.sequence_length+1,0)

                for start_timestep in range (number_of_sequence):
                    self.index.append((demo_name,start_timestep))
    
    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.dataset_path,"r")
        return self._file
    
    @staticmethod
    def _repeat_last_value(value,target_length):
        current_length = value.shape[0]

        if current_length == target_length:
            return value
        
        padding_length = target_length-current_length
        repeat_shape = (padding_length, )+(1,)*(value.ndim-1)
        padding = value[-1:].repeat(repeat_shape)
        return torch.cat((value,padding),0)
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        demo_name, start_timestep = self.index[index]
        demo = self._get_file()["data"][demo_name]

        trajectory_length = demo["actions"].shape[0]

        request_end_timestep = start_timestep+self.sequence_length
        actual_end_timestep = min(request_end_timestep,trajectory_length)

        valid_length = actual_end_timestep-start_timestep
        observation ={}

        for key in self.obs_keys:
            value = torch.as_tensor(demo["obs"][key][start_timestep:actual_end_timestep],dtype=torch.float32)
            if self.pad_sequence:
                value=self._repeat_last_value(value=value,target_length=self.sequence_length)
            observation[key] = value
        actions = torch.as_tensor(demo["actions"][start_timestep:actual_end_timestep],dtype=torch.float32)

        if self.pad_sequence:
            actions = self._repeat_last_value(value=actions,target_length=self.sequence_length)

        sample = {
            "obs":observation,
            "actions":actions,
            "demo_name":demo_name,
            "start_timestep":start_timestep,
            "valid_length": valid_length,
        }

        if self.get_pad_mask:
            pad_mask = torch.zeros(self.sequence_length,1,dtype=torch.float32)
            pad_mask[:valid_length]=1.0
            sample["pad_mask"]=pad_mask
        return sample

    def get_obs_shape(self):
        demo_name,timestep = self.index[0]
        with h5py.File(self.dataset_path,"r") as file:
            demo_obs = file["data"][demo_name]["obs"]

            return {
                key: int(demo_obs[key][timestep].size)
                for key in self.obs_keys
            }

    def get_action_dim(self):
        demo_name,timestep = self.index[0]
        with h5py.File(self.dataset_path,"r") as file:
            actions = file["data"][demo_name]["actions"][timestep]
            return actions.size
    
    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state
    
    def __del__(self):
        self.close()