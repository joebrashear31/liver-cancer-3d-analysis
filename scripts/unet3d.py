
import torch
import torch.nn as nn
import torch.nn.functional as F

# Two 3D convolutions with reLu activation function. max(0, number in the pixel).
# Avoids vanishing gradient - gradient going to 0, causing the model to not make improvements
# SIgmoid and tanh make small gradients, increasing the likelyhood of vanishing gradient.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Gradient descent - would happen using mini-batch - which taskes small 
# batches of the dataset to calculate the loss and update the weights 
# accordingly
#
# All weights are initialized randomly.
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_features=32):
        super(UNet3D, self).__init__()
        self.enc1 = DoubleConv(in_channels, base_features)
        self.enc2 = DoubleConv(base_features, base_features*2)
        self.enc3 = DoubleConv(base_features*2, base_features*4)

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(base_features*4, base_features*8)

        self.up3 = nn.ConvTranspose3d(base_features*8, base_features*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_features*8, base_features*4)

        self.up2 = nn.ConvTranspose3d(base_features*4, base_features*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_features*4, base_features*2)

        self.up1 = nn.ConvTranspose3d(base_features*2, base_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_features*2, base_features)

        self.out_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
