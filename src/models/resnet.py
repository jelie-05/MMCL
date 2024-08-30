import torch
import torchvision.models as models
import torch.nn as nn

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # Freeze


# Define a new model that changes the first conv1 layer based on the mode
class ResNet18_n(nn.Module):
    def __init__(self, n, mode="default", freeze=False):
        super(ResNet18_n, self).__init__()
        self.freeze = freeze

        # Load the pretrained ResNet-18 model
        resnet18 = models.resnet18(pretrained=True)
        set_parameter_requires_grad(resnet18, feature_extracting=self.freeze)  # True: freeze

        if mode == "lidar":
            # Create a new conv1 layer instead of modifying the original
            conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=resnet18.conv1.out_channels,
                kernel_size=resnet18.conv1.kernel_size,
                stride=resnet18.conv1.stride,
                padding=resnet18.conv1.padding,
                bias=False
            )
            # Initialize the weights of the new conv1 layer: USE DEFAULT: KAIMING HE
            # CHECK WHY IT DOESNT WORK WITH THE BELOW INITIALIZATION
            # with torch.no_grad():
            #     conv1.weight = nn.Parameter(resnet18.conv1.weight.sum(dim=1, keepdim=True))
        else:
            # Use the original conv1 layer
            conv1 = resnet18.conv1

        # List of available blocks in ResNet-18
        available_blocks = [resnet18.layer1, resnet18.layer2, resnet18.layer3, resnet18.layer4]

        # Assert that n does not exceed the number of available blocks
        assert n <= len(available_blocks), f"Requested {n} blocks, but only {len(available_blocks)} are available."

        # Extracting the initial convolutional layer, batch norm, and relu
        self.initial_layers = nn.Sequential(
            conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool
        )

        # Collect the blocks iteratively
        blocks = []
        for i in range(n):
            blocks.append(available_blocks[i])

        # Combine the selected blocks into a sequential model
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.blocks(x)
        return x

def resnet18_all_im(mode="default"):
    model = ResNet18_n(n=4, mode=mode)
    return model

def resnet18_all_lid(mode="lidar"):
    model = ResNet18_n(n=4, mode=mode)
    return model

def resnet18_small_im(mode="default"):
    model = ResNet18_n(n=2, mode=mode)
    return model

def resnet18_small_lid(mode="lidar"):
    model = ResNet18_n(n=2, mode=mode)
    return model
