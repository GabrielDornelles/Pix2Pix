import torch
import torch.nn as nn
from torchsummary import summary


class DownBlock(nn.Module):
    """[Conv2d => BatchNorm => LeakyReLU]"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.LeakyReLU(0.2),
        )
       
    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """[TransposedConv2d => BatchNorm => ReLU]"""
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.ReLU()
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):

    def __init__(self, in_channels=3):
        super(Generator, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ) # Same as DownBlock but without batchnorm, as in the paper

        # C64-C128-C256-C512-C512-C512-C512-C512
        self.down1 = DownBlock(in_channels=64, out_channels=128)
        self.down2 = DownBlock(in_channels=128, out_channels=256)
        self.down3 = DownBlock(in_channels=256, out_channels=512)
        self.down4 = DownBlock(in_channels=512, out_channels=512)
        self.down5 = DownBlock(in_channels=512, out_channels=512)
        self.down6 = DownBlock(in_channels=512, out_channels=512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1), 
            nn.ReLU()
        )

        # D512-CD512-CD512-C512-C256-C128-C64
        # up2 to up8 has skip connections, so double the in_channels
        self.up1 = UpBlock(in_channels=512, out_channels=512, use_dropout=True)
        self.up2 = UpBlock(in_channels=512 * 2 , out_channels=512, use_dropout=True)
        self.up3 = UpBlock(in_channels=512 * 2, out_channels=512)
        self.up4 = UpBlock(in_channels=512 * 2, out_channels=512)
        self.up5 = UpBlock(in_channels=512 * 2, out_channels=256)
        self.up6 = UpBlock(in_channels=256 * 2, out_channels=128)
        self.up7 = UpBlock(in_channels=128 * 2, out_channels=64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        # Encoder
        d1 = self.down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bottleneck = self.bottleneck(d7)

        # Decoder
        up1 = self.up1(bottleneck) # 2x2
        up2 = self.up2(torch.cat([up1, d7], 1)) # 4x4
        up3 = self.up3(torch.cat([up2, d6], 1)) # 8x8
        up4 = self.up4(torch.cat([up3, d5], 1)) # 16x16
        up5 = self.up5(torch.cat([up4, d4], 1)) # 32x32
        up6 = self.up6(torch.cat([up5, d3], 1)) # 64x64
        up7 = self.up7(torch.cat([up6, d2], 1)) # 128x128
        up8 = self.up8(torch.cat([up7, d1], 1)) # 256x256
        return up8



if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3)
    outputs = model(x)
    print(f"output size: {outputs.shape}")
    model.to(device=torch.device("cuda"))
    summary(model, (3,256,256))

    

