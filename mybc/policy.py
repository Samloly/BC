import math
import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(
        self,
        obs_shapes,
        obs_keys,
        action_dim,
        hidden_dims=(256,256)
    ):
        super().__init__()

        self.obs_keys = obs_keys
        self.obs_shapes = obs_shapes
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

            layers.append(nn.ReLU())
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
                -1,
            )

            flattened.append(value)
        return torch.cat(
            flattened,
            -1,
        )

    def forward(self, observation):
        observation_vector = (
            self.flatten_observation(observation)
        )

        return self.network(observation_vector)
    
class RNNPolicy(nn.Module):
    def __init__(
        self,
        obs_shapes,
        obs_keys,
        action_dim,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.obs_shapes = obs_shapes
        self.obs_keys = obs_keys
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.observation_dim = sum(
            self._shape_numel(self.obs_shapes[key])
            for key in self.obs_keys
        )

        recurrent_dropout = (self.dropout if self.num_layers>1 else 0.0)
        self.rnn = nn.GRU(
            input_size=self.observation_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=False,
        )

        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.action_dim),
            nn.Tanh(),
        )

    @staticmethod
    def _shape_numel(shape):
        if isinstance(
            shape,
            (tuple, list),
        ):
            return math.prod(shape)

        return int(shape)
    
    def flatten_observation(self,observation):
        flattened_values = []

        for key in self.obs_keys:
            value=observation[key]
            batch_size = value.shape[0]
            sequence_length = value.shape[1]

            value = value.reshape(batch_size,sequence_length,-1)
            flattened_values.append(value)
        
        observation_vector = torch.cat(flattened_values,dim=-1)
        return observation_vector
    
    def initial_hidden_state(self,batch_size,device=None,dtype=None):
        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype
        return torch.zeros(self.num_layers,batch_size,self.hidden_size,device,dtype)
    
    def forward(self,observation,hidden_state=None):
        observation_vector = self.flatten_observation(observation)
        batch_size = observation_vector.shape[0]

        
        rnn_output,next_hidden_state = self.rnn(observation_vector,hidden_state)
        pridicted_actions = self.action_head(rnn_output)

        return pridicted_actions,next_hidden_state

class TransformerPolicy(nn.Module):
    def __init__(
        self, 
        obs_shapes,
        obs_keys,
        action_dim,
        sequence_length=10,
        d_model = 128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()

        self.obs_shapes = obs_shapes
        self.obs_keys = obs_keys
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.observation_dim = sum(self._shape_numel(obs_shapes[key]) for key in self.obs_keys)
        self.obs_embedding = nn.Sequential(
                                nn.Linear(self.observation_dim,d_model),
                                nn.LayerNorm(d_model)
                                )
        self.position_embedding = nn.Parameter(
                                torch.zeros(1,sequence_length,d_model)     
                                )
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
                            encoder_layer=encoder_layer,
                            num_layers=num_layers,
                            norm=nn.LayerNorm(d_model)
                            )
        
        self.action_head = nn.Sequential(
                            nn.Linear(d_model,d_model),
                            nn.GELU(),
                            nn.LayerNorm(d_model),
                            nn.Linear(d_model,self.action_dim),
                            nn.Tanh(),
                            )
        
        self.apply(self._initialize_weights)
        nn.init.normal_(self.position_embedding, mean=0.0,std=0.02)

    @staticmethod
    def _shape_numel(shape):
        if isinstance(shape,(tuple,list)):
            return math.prod(shape)
        return int(shape)
    
    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def flatten_observation(self, observation):
        flattened = []

        for key in self.obs_keys:
            value = observation[key]
            batch_size = value.shape[0]
            sequence_length = value.shape[1]
            value = value.reshape(batch_size,sequence_length,-1)

            flattened.append(value)

        return torch.cat(flattened,-1)
    
    @staticmethod
    def make_causal_mask(sequence_length,device):
        return torch.triu(torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device
            ),
            diagonal=1
        )

    def forward(self,observation,padding_mask=None):
        """
        observation[key]:
            [B, T, *obs_shape]

        padding_mask:
            [B, T]
            True 表示 padding

        returns:
            [B, T, action_dim]
        """
        observation_vector = self.flatten_observation(observation)
        _, sequence_length, _ =observation_vector.shape

        tokens = self.obs_embedding(observation_vector)

        tokens = tokens+self.position_embedding[:,:sequence_length]
        tokens = self.embedding_dropout(tokens)

        causal_mask = self.make_causal_mask(sequence_length,device=tokens.device)
        features = self.transformer(
                        src=tokens,
                        mask=causal_mask,
                        src_key_padding_mask = padding_mask
        )

        predicted_actions = self.action_head(features)

        return predicted_actions