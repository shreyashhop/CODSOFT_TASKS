"""EncoderCNN: pre-trained image feature extractor (ResNet50 or VGG16).

Outputs a spatial grid of features (7x7 -> 49 "pixels") rather than a single
pooled vector, so the decoder can attend over different image regions.
"""

import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, backbone="resnet50", fine_tune=False, fine_tune_layers=2, pretrained=True):
        super().__init__()
        self.backbone_name = backbone

        weights_arg = "DEFAULT" if pretrained else None

        if backbone == "resnet50":
            net = models.resnet50(weights=weights_arg)
            modules = list(net.children())[:-2]  # drop avgpool + fc, keep conv maps
            self.feature_dim = 2048
        elif backbone == "vgg16":
            net = models.vgg16(weights=weights_arg)
            modules = list(net.features)  # conv feature maps only
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.cnn = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.set_fine_tune(fine_tune, fine_tune_layers)

    def forward(self, images):
        """
        images: (batch, 3, H, W)
        returns: (batch, num_pixels=49, feature_dim)
        """
        features = self.cnn(images)                    # (batch, C, h, w)
        features = self.adaptive_pool(features)         # (batch, C, 7, 7)
        features = features.permute(0, 2, 3, 1)         # (batch, 7, 7, C)
        features = features.reshape(features.size(0), -1, features.size(-1))  # (batch, 49, C)
        return features

    def set_fine_tune(self, fine_tune=False, num_layers=2):
        """Freeze the whole CNN by default; optionally unfreeze the last few blocks."""
        for param in self.cnn.parameters():
            param.requires_grad = False

        if fine_tune:
            children = list(self.cnn.children())
            for layer in children[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
