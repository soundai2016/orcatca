"""Acoustic ResNet with a detector and an auxiliary three-class head."""
import math
import torch
from torch import nn

def norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)

class Block(nn.Module):

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), norm(cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, 1, 1, bias=False), norm(cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), norm(cout))

    def forward(self, x):
        return torch.nn.functional.silu(self.main(x) + self.skip(x))

class AcousticNet(nn.Module):

    def __init__(self, width=32, dropout=0.15):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, width, 7, 2, 3, bias=False), norm(width), nn.SiLU(), nn.MaxPool2d(3, 2, 1))
        blocks = []
        cin = width
        for i in range(4):
            cout = width * 2 ** i
            blocks += [Block(cin, cout, 1 if i == 0 else 2), Block(cout, cout, 1)]
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin * 2, 5))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.blocks(self.stem(x)).flatten(2)
        mean = x.mean(-1)
        std = (x.var(-1, unbiased=False) + 1e-05).sqrt()
        return self.head(torch.cat((mean, std), dim=1))
