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

# class RobomimicDataset(Dataset):
#     def __init__(self,dataset_path,split,obs_keys):
#         self.dataset_path = dataset_path
#         self.obs_key = tuple(obs_keys)
#         self.split = split
#         self.index = []
#         self._file = None

#         with h5py.File(self.dataset_path,"r") as file:
#             demo_names = [
#                 name.decode() if isinstance(name,bytes) else str(name)
#                 for name in file["mask"][split][:]
#             ]

#             for demo_name in demo_names:
#                 trajectory_length = file["data"][demo_name]["actions"].shape[0]

#                 for timestep in range(trajectory_length):
#                     self.index.append((demo_name,timestep))

#         self._file = None

#     def _get_file(self):
#         if self._file is None:
#             self._file = h5py.File(self.dataset_path,"r")

#         return self._file
    
#     def __len__(self):
#         return len(self.index)
    
#     def __getitem__(self,index):
#         demo_name, timestep = self.index[index]
#         demo = self._get_file()["data"][demo_name]

#         observation = {
#             key:torch.as_tensor(
#                 demo["obs"][key][timestep],
#                 dtype=torch.float32,
#             )
#             for key in self.obs_key
#         }

#         actions = torch.as_tensor(
#             demo["actions"][timestep],
#             dtype=torch.float32,
#         )

#         return {
#             "obs":observation,
#             "actions":actions,
#             "demo_name":demo_name,
#             "timestep":timestep,
#         }
    
#     def get_obs_shape(self):
#         demo_name,timestep = self.index[0]

#         with h5py.File(self.dataset_path,"r") as file:
#             demo_obs = file["data"][demo_name]["obs"]
#             return {
#                 key: int(
#                     demo_obs[key][timestep].size
#                 )
#                 for key in self.obs_key
#             }
        
#     def get_action_dim(self):
#         demo_name,timestep = self.index[0]

#         with h5py.File(self.dataset_path,"r") as file:
#             actions = file["data"][demo_name]["actions"][timestep]
#             return int(actions.size)
    

#     def close(self):
#         if self._file is not None:
#             self._file.close()
#             self._file = None
    
#     def __getstate__(self):
#         state = self.__dict__.copy()
#         state["_file"] = None
#         return state
    
#     def __del__(self):
#         self.close()
#         # super().__init__()

class RobomimicDataset(Dataset):
    def __init__(self,dataset_path,split,low_dim_keys,rgb_keys=()):
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.low_dim_keys = low_dim_keys
        self.rgb_keys = rgb_keys
        self.index=[]
        self._file = None

        with h5py.File(self.dataset_path,"r") as file:
            demo_names = decode_demo_names(file["mask"][split][:])

            for demo_name in demo_names:
                trajectory_length = file["data"][demo_name]["actions"].shape[0]
                for timestep in range(trajectory_length):
                    self.index.append((demo_name,timestep))

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.dataset_path,"r")

        return self._file
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        demo_name, timestep = self.index[index]
        demo = self._get_file()["data"][demo_name]
        obs_group = demo["obs"]

        observation = {}

        for key in self.low_dim_keys:
            observation[key] = torch.as_tensor(
                obs_group[key][timestep],
                dtype=torch.float32
            )

        for key in self.rgb_keys:
            image = torch.as_tensor(
                obs_group[key][timestep],
                dtype=torch.float32
            )
            # (H, W, C) → (C, H, W)
            image = image.permute(2,0,1)

            # [0,255] → [0,1]
            observation[key] = image/255.0
            
        actions = torch.as_tensor(
            demo["actions"][timestep],
            dtype=torch.float32,
        )

        return {
            "obs":observation,
            "actions": actions
        }

    def get_obs_shapes(self):
        demo_name,_ = self.index[0]

        with h5py.File(self.dataset_path,"r") as file:
            obs = file["data"][demo_name]["obs"]
            shapes ={}

            for key in self.low_dim_keys:
                shapes[key] = obs[key].shape[1:]

            for key in self.rgb_keys:
                height,width,channels = obs[key].shape[1:]
                shapes[key] = (channels,height,width)
            
            return shapes
        
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

class ACTRobomimicDataset(RobomimicDataset):
    def __init__(self, dataset_path, split, low_dim_keys, rgb_keys,chunk_size=20):
        super().__init__(dataset_path, split, low_dim_keys, rgb_keys)

        self.chunk_size = chunk_size

    @staticmethod
    def _read_rgb_image(image_dataset,timestep):
        image = torch.as_tensor(image_dataset[timestep],dtype=torch.float32)

        if image.shape[-1] in (3,4):
            image = image.permute(2,0,1)
        if image.shape[0]==4:
            image = image[:3]

        if image.max()>1.0:
            image = image/255.0

        return image.contiguous()
    
    def __getitem__(self, index):
        demo_name,timestep = self.index[index]
        demo = self._get_file()["data"][demo_name]
        obs_group = demo["obs"]
        observation = {}
        
        for key in self.low_dim_keys:
            observation[key] = torch.as_tensor(
                obs_group[key][timestep],
                dtype=torch.float32
            )

        for key in self.rgb_keys:
            observation[key] = self._read_rgb_image(obs_group[key],timestep)

        trajectory_length = demo["actions"].shape[0]

        requested_end = timestep+self.chunk_size

        actual_end = min(requested_end,trajectory_length)

        valid_length = actual_end-timestep

        future_actions = torch.as_tensor(demo["actions"][timestep:actual_end],dtype=torch.float32)

        action_dim = demo["actions"].shape[-1]

        actions = torch.zeros(self.chunk_size,action_dim,dtype=torch.float32)
        actions[:valid_length] = future_actions
        #false表示有效，True表示padding
        is_pad = torch.ones(self.chunk_size,dtype=torch.bool)
        is_pad[:valid_length] = False

        return {
            "obs":observation,
            "actions":actions,
            "is_pad":is_pad,
            "demo_name":demo_name,
            "timestep":timestep,
            "valid_length":valid_length
        }

