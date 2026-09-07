import math
import torch
import torch.nn as nn

from torchvision.models import(
    ResNet18_Weights,
    resnet18,
)

class ResNet18SpatialBackbone(nn.Module):
    """
    ImageNet-pretrained ResNet18 truncated
    after layer3.

    Input:
        images: [B, 3, H, W]
        Expected range: [0, 1]

    Output:
        feature_map:
            [B, 256, H_feature, W_feature]

        For 84x84 input:
            [B, 256, 6, 6]
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # weights = ResNet18_Weights.DEFAULT if pretrained else None
        default_weights = ResNet18_Weights.DEFAULT
        model_weights = (
            default_weights
            if pretrained
            else None
        )

        model = resnet18(weights=model_weights)

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        
        self.output_channels = 256

        preprocessing = default_weights.transforms()
        image_mean = preprocessing.mean
        image_std = preprocessing.std

        self.register_buffer("image_mean",torch.tensor(image_mean,dtype=torch.float32).view(1,3,1,1))
        self.register_buffer("image_std",torch.tensor(image_std,dtype=torch.float32).view(1,3,1,1))
    
    def normalize_images(self,images):
        return (images-self.image_mean)/self.image_std
    
    def forward(self, images):
        images = self.normalize_images(images)

        features = self.stem(images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)

        return features
    
def build_2d_sincos_position_embedding(height,width,embedding_dim,device,dtype):
    """
    Return:
        [height * width, embedding_dim]
    """
    y_positions = torch.linspace(0.0,1.0,steps=height,device=device,dtype=dtype)
    x_positions = torch.linspace(0.0,1.0,steps=width,device=device,dtype=dtype)

    grid_y,grid_x = torch.meshgrid(y_positions,x_positions,indexing="ij")

    grid_x = grid_x.reshape(-1,1)*2.0*math.pi
    grid_y = grid_y.reshape(-1,1)*2.0*math.pi

    quarter_dim = embedding_dim//4

    frequencies = torch.arange(quarter_dim,device=device,dtype=dtype)
    if quarter_dim>1:
        frequencies = frequencies/(quarter_dim-1)

    frequencies=1.0/(10000.0**frequencies)

    x_angles = grid_x*frequencies.unsqueeze(0)
    y_angles = grid_y*frequencies.unsqueeze(0)

    position_embedding = torch.cat(
        [
            torch.sin(x_angles),
            torch.cos(x_angles),
            torch.sin(y_angles),
            torch.cos(y_angles),
        ],
        dim=-1,
    )

    return position_embedding

class MultiCameraSpatialEncoder(nn.Module):
    def __init__(self, camera_keys,d_model=256,pretrained=True):
        super().__init__()

        self.camera_keys = camera_keys
        self.d_model = d_model
        self.backbone = ResNet18SpatialBackbone(pretrained=pretrained)
        self.feature_projection = nn.Conv2d(
            in_channels=self.backbone.output_channels,
            out_channels=self.d_model,
            kernel_size=1,
        )

        self.position_encoder = PositionEmbeddingSine(
            embedding_dim=self.d_model,
            temperature=10000,
            normalize=True,
        )
        


    def forward(self,observation):
        images_by_camera = []
        batch_size = None
        image_shape = None

        for key in self.camera_keys:
            images = observation[key]
            if batch_size is None:
                batch_size = images.shape[0]
                image_shape = images.shape[1:]

            images_by_camera.append(images)

        #[B,C,3,H,W]
        stacked_images = torch.stack(images_by_camera,dim=1)

        number_of_cameras = len(self.camera_keys)

        _,_,channels,height,width = stacked_images.shape

        #[B*C,3,H,W]
        flat_images = stacked_images.reshape(batch_size*number_of_cameras,channels,height,width)

        #[B*C,256,Hf,Wf]
        feature_maps = self.backbone(flat_images)

        #[B*C.d_model,Hf,Wf]
        feature_maps = self.feature_projection(feature_maps)

        position_maps = self.position_encoder(feature_maps)

        _,_,feature_height,feature_width = feature_maps.shape

        feature_maps = feature_maps.reshape(
            batch_size,
            number_of_cameras,
            self.d_model,
            feature_height,
            feature_width
        )

        position_maps =position_maps.reshape(
            batch_size,
            number_of_cameras,
            self.d_model,
            feature_height,
            feature_width
        )

        features_by_camera = [
            feature_maps[:,camera_index]
            for camera_index in range(number_of_cameras)
        ]

        positions_by_camera = [
            position_maps[:,camera_index]
            for camera_index in range(number_of_cameras)
        ]

        feature_maps = torch.cat(features_by_camera,dim=3)
        position_maps = torch.cat(positions_by_camera,dim=3)

        # number_of_spatial_tokens = feature_height*feature_width

        # [B*C,Hf*Wf,d_model]
        visual_features = feature_maps.flatten(start_dim=2).transpose(1,2)

        visual_positions = position_maps.flatten(start_dim=2).transpose(1,2)
        
        return {
            "features": visual_features,
            "positions":visual_positions
        }
    
class PositionEmbeddingSine(nn.Module):
    """
    DETR-style two-dimensional sine/cosine
    position embedding.

    Input:
        feature_map:
            [B, C, H, W]

        mask:
            Optional [B, H, W].
            True means padded/invalid position.

    Output:
        position_embedding:
            [B, d_model, H, W]
    """
    def __init__(
        self,
        embedding_dim=256,
        temperature=10000,
        normalize=True,
        scale=None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_position_features = embedding_dim//2

        self.temperature = temperature 
        self.normalize = normalize

        self.scale = 2.0*math.pi if scale is None else scale

    def forward(self,feature_map,mask=None):
        """
            feature_map: [B,C,H,W]
            position:[B,D,H,W]
        """
        batch_size,_,height,width = feature_map.shape
        # [B,H,W]
        if mask is None:
            mask = torch.zeros(
                batch_size,
                height,
                width,
                dtype=torch.bool,
                device=feature_map.device
            )
        
        valid_mask = ~mask
        #纵向累加
        y_embedding = valid_mask.cumsum(dim=1,dtype=torch.float32)
        #横向累加
        x_embedding = valid_mask.cumsum(dim=2,dtype=torch.float32)
        # 归一化到[0,2pi]
        if self.normalize:
            epsilon = 1e-6
            y_embedding=y_embedding/(y_embedding[:,-1:,:]+epsilon)*self.scale
            x_embedding = x_embedding/(x_embedding[:,:,-1:]+epsilon)*self.scale
        
        dimension = torch.arange(self.num_position_features,dtype=torch.float32,device=feature_map.device)

        dimension = self.temperature ** (
            2
            * torch.div(
                dimension,
                2,
                rounding_mode="floor",
            )
            / self.num_position_features
        )

        position_x = x_embedding[:,:,:,None]/dimension
        position_y = y_embedding[:,:,:,None]/dimension

        position_x = torch.stack(
            (
                position_x[:, :, :, 0::2].sin(),
                position_x[:, :, :, 1::2].cos(),
            ),
            dim=4,
        ).flatten(3)

        position_y = torch.stack(
            (
                position_y[:, :, :, 0::2].sin(),
                position_y[:, :, :, 1::2].cos()
            ),
            dim=4
        ).flatten(3)

        position = torch.cat(
            (
                position_y,position_x
            ),
            dim=3
        )

        # [B, H, W, D] -> [B, D, H, W]
        position = position.permute(0,3,1,2)
        return position.to(dtype=feature_map.dtype)