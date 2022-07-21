import torch
import torch.nn as nn
from torchsummary import summary

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # discriminator takes both images, we concat them so we need in_channels * 2
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ) # same as generator, block without batchnorm

        self.block1 = Block(in_channels=64, out_channels=128, stride=2)
        self.block2 = Block(in_channels=128, out_channels=256, stride=2)
        self.block3 = Block(in_channels=256, out_channels=512, stride=1)
        self.last_conv = nn.Conv2d(512, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
      

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.first_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.last_conv(x)
        return x


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    outputs = model(x, y)
    print(f"output size: {outputs.shape}")
    model.to(device=torch.device("cuda"))
    summary(model, [(3,256,256),(3,256,256)])
