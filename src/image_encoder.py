# SPDX-License-Identifier: Apache-2.0

from torchvision.models import mobilenet_v2
import torch
import torch.nn as nn

class ImageEncoderBackbone(nn.Module):
    """Class for image encoder 

    Parameters
    ----------
    hidden_size : int
        Embedding size.
    image_model : string
        Model name to load.
    pretrained : bool
        Whether to load pretrained imagenet weights.

    """
    def __init__(self, hidden_size=1024, image_model='mobilenet_v2'):
        super(ImageEncoderBackbone, self).__init__()
        self.image_model = image_model
        backbone = globals()[image_model](pretrained=False)
        if 'mobilenet' in image_model:
            modules = list(backbone.children())[:-1]
            in_feats = backbone.classifier[1].in_features
        else:
            modules = list(backbone.children())[:-2]
            in_feats = backbone.fc.in_features

        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(in_feats, hidden_size)

    def forward(self, images, freeze_backbone=False):
        """Extract feature vectors from input images."""
        if not freeze_backbone:
            feats = self.backbone(images)
        else:
            with torch.no_grad():
                feats = self.backbone(images)
        feats = feats.view(feats.size(0), feats.size(1),
                           feats.size(2)*feats.size(3))

        feats = torch.mean(feats, dim=-1)
        out = self.fc(feats)

        return nn.Tanh()(out)



class ImageEmbedding(nn.Module):
    """ Extract embedding of images
    Parameters
    ----------
    output_size : int
        Embedding output size.
    image_model : string
        Name of image model.
    """
    def __init__(self, output_size, image_model):
        super(ImageEmbedding, self).__init__()
        self.image_encoder = ImageEncoderBackbone(output_size, image_model)

    def forward(self, img, freeze_backbone=True):
        img_feat = self.image_encoder(img, freeze_backbone=freeze_backbone)
        return img_feat