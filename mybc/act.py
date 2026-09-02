import torch
import torch.nn as nn

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
    ):
        super().__init__()

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.observation_encoder = ACTObservationEncoder(
            low_dim_keys=low_dim_keys,
            camera_keys=camera_keys,
            obs_shapes=obs_shapes,
            d_model=self.d_model,
            pretrained_backbone=pretrained_backbone
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

    def forward(self,observation):
        memory_tokens = self.observation_encoder(observation)
        batch_size = memory_tokens.shape[0]
        action_queries = self.action_queries.unsqueeze(0).expand(batch_size,-1,-1)

        action_features = self.decoder(
            tgt = action_queries,
            memory = memory_tokens
        )

        predicted_actions = self.action_head(action_features)

        return predicted_actions