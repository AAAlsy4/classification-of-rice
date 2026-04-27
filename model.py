import torch
from torch import nn
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1conv=False, stride=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        out = self.ReLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            x = self.conv3(x)
        out += x
        out = self.ReLU(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, Residual):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(
            Residual(64, 64),
            Residual(64, 64)
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, stride=2),
            Residual(128, 128)
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, stride=2),
            Residual(256, 256)
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, stride=2),
            Residual(512, 512)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 5)
        )

        # 不需要初始化权重，因为使用了BatchNorm2d，模型会自动初始化

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(Residual).to(device)
    print(summary(model,(1,224,224)))