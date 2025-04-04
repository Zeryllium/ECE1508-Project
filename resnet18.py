import torch
import torch.nn as nn


def _initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Handle downsampling and if input channels != output channels for the res block skip connection
        # (i.e. every other resnetblock)
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_connection = nn.Sequential()

    def forward(self, x):
        z = self.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        # Handle skip connection with dissimilar in channels and out channels
        z += self.skip_connection(x)
        y = self.relu(z)
        return y


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.block1 = ResNetBlock(64, 64)
        self.block2 = ResNetBlock(64, 64)

        self.block3 = ResNetBlock(64, 128, stride=2)
        self.block4 = ResNetBlock(128, 128)

        self.block5 = ResNetBlock(128, 256, stride=2)
        self.block6 = ResNetBlock(256, 256)

        self.block7 = ResNetBlock(256, 512, stride=2)
        self.block8 = ResNetBlock(512, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # CIFAR-10 classes
        self.fc = nn.Linear(512, 10)

        self.apply(_initialize_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)

        x = self.block5(x)
        x = self.block6(x)

        x = self.block7(x)
        x = self.block8(x)

        x = self.avgpool(x)
        x = self.fc(x.squeeze())

        return x

