import math
import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(
        self,
        obs_shpaes,
        obs_keys,
        action_dim,
        hidden_dims=(256,256)
    ):
        super().__init__()

        self.obs_keys = obs_keys
        self.obs_shapes = obs_shpaes
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        observation_dim = sum(
            math.prod(
                (
                    self.obs_shapes[key]
                    if isinstance(
                        self.obs_shapes[key],
                        (tuple, list),
                    )
                    else (self.obs_shapes[key],)
                )
            )
            for key in self.obs_keys
        )

        layers = []
        input_dim = observation_dim

        for hidden_dim in self.hidden_dims:
            layers.append(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                )
            )

            layers.append(nn.Relu())
            input_dim = hidden_dim
        
        layers.append(
            nn.Linear(
                input_dim,
                self.action_dim,
            )
        )

        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def flatten_observation(
        self,
        observation,
    ):
        flattened = []

        for key in self.obs_keys:
            value = observation[key]
            value = value.reshape(
                value.shape[0],
                dim=-1,
            )

            flattened.append(value)
        return torch.cat(
            flattened,
            dim=-1,
        )

    def forward(self, observation):
        observation_vector = (
            self.flatten_observation(observation)
        )

        return self.network(observation_vector)