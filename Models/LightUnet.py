import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """ Double Convolution """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    """ UNET architecture """

    def __init__(self, in_channels, out_channels):
        super(UNET, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.down1 = DoubleConv(in_channels, 64)
        self.down1 = DoubleConv(in_channels, 128)
        # self.down2 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        # self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.conv4 = DoubleConv(128, 64)

        # self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        # x4 = self.down4(self.pool(x3))

        # bottleneck = self.bottleneck(self.pool(x4))
        bottleneck = self.bottleneck(self.pool(x3))

        x = self.conv1(torch.cat([x3, self.up1(bottleneck)], dim=1))
        x = self.conv2(torch.cat([x2, self.up2(x)], dim=1))
        x = self.conv3(torch.cat([x1, self.up3(x)], dim=1))
        # x = self.conv4(torch.cat([x1, self.up4(x)], dim=1))

        x = self.final(x)
        return x


if __name__ == "__main__":
    batch_size = 2
    in_channels = 3
    num_classes = 5
    img_height = img_width = 256  # only works for divisible with 16

    x = torch.randn((batch_size, in_channels, img_height, img_width))
    model = UNET(in_channels=in_channels, out_channels=num_classes)
    out = model(x)

    print('Image shape:      :', x.shape)
    print('Prediction shape: :', out.shape)