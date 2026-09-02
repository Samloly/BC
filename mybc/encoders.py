import math
import torch
import torch.nn as nn

class ProprioceptionEncoder(nn.Module):
    """
    Convert multiple low-dimensional
    observations into one Transformer token.

    Input:
        observation[key]:
            [B, *obs_shape]

    Output:
        token:
            [B, 1, d_model]
    """
    def __init__(self, low_dim_keys, obs_shapes,d_model=256):
        super().__init__()

        self.low_dim_keys = low_dim_keys
        self.d_model = d_model
        self.input_dim = sum(math.prod(obs_shapes[key]) for key in self.low_dim_keys)

        self.network = nn.Sequential(
            nn.Linear(self.input_dim,self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model,self.d_model)
        )

    def forward(self,observation):
        flattened_values = []
        batch_size = None

        for key in self.low_dim_keys:
            value = observation[key]
            current_batch_size = value.shape[0]
            if batch_size is None:
                batch_size = current_batch_size
            
            value = value.reshape(current_batch_size,-1)

            flattened_values.append(value)

        Proprioception = torch.cat(flattened_values,dim=-1)

        token =self.network(Proprioception)
        return token.unsqueeze(1)