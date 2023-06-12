import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms.functional as TF
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1):
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        super(AttU_Net, self).__init__()

        # filters = [64, 128, 256, 512, 1024]


        self.encoder0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu)

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        # self.encoder4 = backbone.layer4

        self.maxpool = backbone.maxpool
        self.Up5 = up_conv(1024, 512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(1024, 512)

        self.Up4 = up_conv(512, 256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(512, 256)

        self.Up3 = up_conv(256, 128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(256, 128)

        self.Up2 = up_conv(128, 64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.encoder0(x)  # 64 0
        e1 = self.maxpool(e1)
        print(e1.shape)
        # print(e2.shape)
        e2 = self.encoder1(e1)  # 128 1
        print(e2.shape)
        e3 = self.encoder2(e2)  # 256 2
        print(e3.shape)
        e4 = self.encoder3(e3)  # 512 3
        print(e4.shape)


        d5 = self.Up5(e4)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)  # 512

        d4 = self.Up4(d5)  # 256
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)  # 256

        d3 = self.Up3(d4)  # 128
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)  # 128

        d2 = self.Up2(d3)  # 64
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)  # 64

        out = self.Conv(d2)

        return out


def main():
    x = torch.randn((16, 3, 256, 512))
    model = AttU_Net()
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    # assert preds.shape == x.shape


if __name__ == "__main__":
    main()