import torch
import torch.nn as nn
import math
from mybc.encoders import ProprioceptionEncoder
from mybc.vision import MultiCameraSpatialEncoder


class ACTObservationEncoder(nn.Module):
    """
    Encode RGB and proprioception into
    Transformer memory tokens.
    """
    def __init__(
        self,
        low_dim_keys,
        camera_keys,
        obs_shapes,
        d_model=256,
        pretrained_backbone=True
    ):
        super().__init__()

        self.visual_encoder = MultiCameraSpatialEncoder(
            camera_keys=camera_keys,
            d_model = d_model,
            pretrained=pretrained_backbone,
        )

        self.proprio_encoder = ProprioceptionEncoder(
            low_dim_keys=low_dim_keys,
            obs_shapes=obs_shapes,
            d_model=d_model,
        )

    def forward(self,observation):
        proprio_token = self.proprio_encoder(observation)
        visual_tokens = self.visual_encoder(observation)

        memory_toekns = torch.cat(
            [
                proprio_token,
                visual_tokens,
            ],
            dim=1
        )

        return memory_toekns
    
class ACTPolicy(nn.Module):
    def __init__(
        self,
        low_dim_keys,
        camera_keys,
        obs_shapes,
        action_dim,
        chunk_size=20,
        d_model=256,
        nhead=8,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pretrained_backbone=True,
        latent_dim=32,
        num_latent_encoder_layers=4
    ):
        super().__init__()

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.low_dim_keys = low_dim_keys
        self.latent_dim = latent_dim
        self.observation_encoder = ACTObservationEncoder(
            low_dim_keys=low_dim_keys,
            camera_keys=camera_keys,
            obs_shapes=obs_shapes,
            d_model=self.d_model,
            pretrained_backbone=pretrained_backbone
        )

        proprop_dim =sum(self._shape_numel(obs_shapes[key]) for key in self.low_dim_keys)
        self.proprio_dim = proprop_dim

        self.latent_encoder = ACTLatentEncoder(
            proprio_dim=self.proprio_dim,
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            d_model=self.d_model,
            latent_dim=self.latent_dim,
            nhead=nhead,
            num_layers=num_latent_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.latent_to_token = nn.Sequential(
            nn.Linear(self.latent_dim,self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.action_queries = nn.Parameter(
            torch.empty(
                self.chunk_size,
                self.d_model
            )
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(self.d_model)
        )

        self.action_head = nn.Linear(self.d_model,self.action_dim)

        nn.init.normal_(self.action_queries,mean=0.0,std=0.02)

    @staticmethod
    def _shape_numel(shape):
        if isinstance(shape,(tuple,list)):
            return math.prod(shape)
        return shape
    
    def concatenate_proprioception(self,observation):
        """
        Concatenate normalized low-dimensional
        observations.

        Input:
            observation[key]:
                [B, *obs_shape]

        Output:
            proprioception:
                [B, proprio_dim]
        """
        values = []
        batch_size = None

        for key in self.low_dim_keys:
            value = observation[key]
            current_batch_size = value.shape[0]
            if batch_size is None:
                batch_size = current_batch_size
            
            value = value.reshape(current_batch_size,-1)
            values.append(value)
        
        proprioception = torch.cat(values,dim=-1)
        return proprioception

    def forward(self,observation,actions=None,is_pad=None):
        memory_tokens = self.observation_encoder(observation)
        batch_size = memory_tokens.shape[0]
        proprioception = self.concatenate_proprioception(observation)

        if actions is not None:
            z,mu,logvar = self.latent_encoder(
                proprioception=proprioception,
                actions=actions,
                is_pad=is_pad
            )
        
        else:
            z=torch.zeros(
                batch_size,
                self.latent_dim,
                device=memory_tokens.device,
                dtype=memory_tokens.dtype
            )
            mu=None
            logvar=None
        
        latent_token =self.latent_to_token(z).unsqueeze(1)
        memory_tokens=torch.cat(
            [
                latent_token,
                memory_tokens
            ],
            dim=1
        )

        action_queries = self.action_queries.unsqueeze(0).expand(batch_size,-1,-1)

        action_features = self.decoder(
            tgt = action_queries,
            memory = memory_tokens
        )

        predicted_actions = self.action_head(action_features)

        return {
            "predicted_actions":predicted_actions,
            "mu":mu,
            "logvar":logvar
        }
    
class ACTLatentEncoder(nn.Module):
    """
    Encode current proprioception and an
    expert action chunk into a CVAE latent.

    Inputs:
        proprioception: [B, proprio_dim]
        actions:        [B, K, action_dim]
        is_pad:         [B, K]

    Outputs:
        z:              [B, latent_dim]
        mu:             [B, latent_dim]
        logvar:         [B, latent_dim]
    """
    def __init__(
        self,
        proprio_dim,
        action_dim,
        chunk_size,
        d_model=256,
        latent_dim=32,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()

        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.proprio_projection = nn.Linear(self.proprio_dim,self.d_model)
        self.action_projection = nn.Linear(self.action_dim,self.d_model)

        self.cls_token = nn.Parameter(torch.empty(1,1,self.d_model))

        self.position_embedding = nn.Parameter(torch.empty(1,self.chunk_size+2,self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.d_model)
        )

        self.latent_projection = nn.Linear(self.d_model,2*self.latent_dim)

        nn.init.normal_(self.cls_token,mean=0.0,std=0.02)
        nn.init.normal_(self.position_embedding,mean=0.0,std=0.02)

    def forward(self,proprioception,actions,is_pad):
        batch_size = actions.shape[0]
        cls_token = self.cls_token.expand(batch_size,-1,-1)
        proprio_token = self.proprio_projection(proprioception).unsqueeze(1)

        action_tokens = self.action_projection(actions)

        tokens =torch.cat(
            [
                cls_token,
                proprio_token,
                action_tokens
            ],
            dim=1
        )
        tokens = tokens+self.position_embedding

        prefix_pad =torch.zeros(batch_size,2,dtype=torch.bool,device=is_pad.device)
        padding_mask=torch.cat(
            [
                prefix_pad,
                is_pad,
            ],
            dim=1
        )

        encoded = self.transformer_encoder(
            src=tokens,
            src_key_padding_mask = padding_mask
        )

        cls_feature = encoded[:,0]

        latent_parameters = self.latent_projection(cls_feature)

        mu,logvar = torch.chunk(latent_parameters,chunks=2,dim=-1)

        std = torch.exp(0.5*logvar)

        epsilon = torch.randn_like(std)

        z=mu+epsilon*std

        return z,mu,logvar

