import torch
import torch.nn as nn
from torchvision.models import resnet18
from inference.train.mmsiamese.calc_receptive_field import PixelwiseFeatureMaps

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # Freeze


class classifier_head(nn.Module):
    def __init__(self, model_im, model_lid, pixel_wise, masking):
        super().__init__()
        self.model_im = model_im
        self.model_lid = model_lid
        self.pixel_wise = pixel_wise
        self.masking = masking
        set_parameter_requires_grad(self.model_im, feature_extracting=True)
        set_parameter_requires_grad(self.model_lid, feature_extracting=True)

        self.classifier_layers = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # output: (N, 512, 12, 39)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # output: (N, 512, 6, 20)
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),  # output: (N, 1024, 3, 10)
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # output: (N, 1024, 1, 1)
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
            )

    def load_classifier_layers(self, state_dict):
        classifier_state_dict = {k.replace('classifier_layers.', ''): v for k, v in state_dict.items() if
                                 k.startswith('classifier_layers.')}
        self.classifier_layers.load_state_dict(classifier_state_dict)

    def forward(self, image, lidar, H, W):
        pred_im = self.model_im(image)
        pred_lid = self.model_lid(lidar)
        if self.pixel_wise:
                pixel_im = PixelwiseFeatureMaps(model=self.model_im, embeddings_value=pred_im,
                                                input_image_size=(H, W))
                pred_im = pixel_im.assign_embedding_value()
                pixel_lid = PixelwiseFeatureMaps(model=self.model_lid, embeddings_value=pred_lid,
                                                 input_image_size=(H, W))
                pred_lid = pixel_lid.assign_embedding_value()

        # concatenate x, y as z
        z = torch.cat((pred_im, pred_lid), dim=1)
        z = self.classifier_layers(z)
        return z