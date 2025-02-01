import torch.nn as nn
import timm


class AgeGenderModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        feature_size = self.backbone.num_features

        # Age regression head (0-116 years)
        self.age_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Constrain output to 0-1 range
        )

        # Gender classification head with sigmoid
        self.gender_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features) * 116  # Scale to 0-116
        gender = self.gender_head(features)
        return age, gender