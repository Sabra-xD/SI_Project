import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64); self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128); self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256); self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2); self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2); self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2); self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2); self.conv1 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_channels, 1)
    def forward(self, x):
        d1, p1 = self.down1(x), self.pool1(self.down1(x))
        d2, p2 = self.down2(p1), self.pool2(self.down2(p1))
        d3, p3 = self.down3(p2), self.pool3(self.down3(p2))
        d4, p4 = self.down4(p3), self.pool4(self.down4(p3))
        bn = self.bottleneck(p4)
        u4 = self.up4(bn); c4 = self.conv4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(c4); c3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(c3); c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2); c1 = self.conv1(torch.cat([u1, d1], dim=1))
        return torch.sigmoid(self.outc(c1))