import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        res_conv = self.conv(x)
        res_pool = self.pool(res_conv)

        return res_conv, res_pool

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=2, stride=2)
        self.conv = ConvBlock(2 * out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        dH = skip.size()[2] - x.size()[2]
        dW = skip.size()[3] - x.size()[3]

        x = nn.functional.pad(x, [dW // 2, dW - dW // 2,
                                  dH // 2, dH - dH // 2])

        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        '''
        Initializes the U-Net model, defining the encoder, decoder, and other layers.

        Args:
        - in_channels (int): Number of input channels (1 for scan images).
        - out_channels (int): Number of output channels (1 for binary segmentation masks).
        
        Function:
        - CBR (in_channels, out_channels): Helper function to create a block of Convolution-BatchNorm-ReLU layers. 
        (This function is optional to use)
        '''
        super(UNet, self).__init__()

        # encoder
        self.maxpool = nn.MaxPool2d(2)
        self.encoder1 = DownSampleBlock(in_channels, 64)
        self.encoder2 = DownSampleBlock(64, 128)
        self.encoder3 = DownSampleBlock(128, 256)
        self.encoder4 = DownSampleBlock(256, 512)

        # bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # decoder
        self.upconv1 = UpSampleBlock(1024, 512)
        self.upconv2 = UpSampleBlock(512, 256)
        self.upconv3 = UpSampleBlock(256, 128)
        self.upconv4 = UpSampleBlock(128, 64)

        # n_channel output
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        '''
        Defines the forward pass of the U-Net, performing encoding, bottleneck, and decoding operations.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        '''
        x1, x = self.encoder1(x)
        x2, x = self.encoder2(x)
        x3, x = self.encoder3(x)
        x4, x = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.upconv1(x, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.upconv4(x, x1)

        x = self.out(x)
        return x