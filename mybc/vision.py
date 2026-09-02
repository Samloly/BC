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

        self.camera_embedding = nn.Embedding(
            num_embeddings=len(self.camera_keys),
            embedding_dim=self.d_model,
        )

        nn.init.normal_(self.camera_embedding.weight,mean=0.0,std=0.02)

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

        _,_,feature_height,feature_width = feature_maps.shape

        number_of_spatial_tokens = feature_height*feature_width

        # [B*C,Hf*Wf,d_model]
        tokens = feature_maps.flatten(start_dim=2).transpose(1,2)
        
        #[B,C,Hf*Wf,d_model]
        tokens = tokens.reshape(batch_size,number_of_cameras,number_of_spatial_tokens,self.d_model)

        position_embedding = build_2d_sincos_position_embedding(
            height=feature_height,
            width=feature_width,
            embedding_dim=self.d_model,
            device=tokens.device,
            dtype=tokens.dtype
        )

        #[1，1，H*fWf,d_model]
        position_embedding = position_embedding.view(1,1,number_of_spatial_tokens,self.d_model)

        #[1,C,1,d_model]
        camera_embedding = self.camera_embedding.weight.view(1,number_of_cameras,1,self.d_model)

        tokens = tokens+position_embedding+camera_embedding

        tokens = tokens.reshape(batch_size,number_of_cameras*number_of_spatial_tokens,self.d_model)
        return tokens