import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * reduction),  # //意思是整数除法
            nn.ReLU(inplace=True),
            nn.Linear(channel * reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()  # .shape属性和.size()方法是一个东西
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return y


class SEBlock_(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock_, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # //意思是整数除法
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # .shape属性和.size()方法是一个东西
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class ResidualBlock2D_(nn.Module):

    def __init__(self, planes, kernel_size=(11, 5), padding=(5, 2), downsample=True):
        super(ResidualBlock2D_, self).__init__()
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes * 2, kernel_size=kernel_size, stride=1,
                            padding=padding, bias=False)
        self.b2 = nn.BatchNorm2d(planes * 2)
        self.c3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=1, bias=False)
        self.downsample = downsample
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)
        # print(out.shape, identity.shape)
        out += identity
        out = self.relu(out)

        return out


class ResidualBlock1D_(nn.Module):

    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D_, self).__init__()
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes, planes * 2, kernel_size=11, stride=1,
                            padding=5, bias=False)
        self.b2 = nn.BatchNorm1d(planes * 2)
        self.c3 = nn.Conv1d(planes * 2, planes * 8, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm1d(planes * 8)
        self.downsample = nn.Sequential(
            nn.Conv1d(planes, planes * 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes * 8),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualBlock1D(nn.Module):
    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D, self).__init__()
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes, planes, kernel_size=11, stride=1,  # kernel 11
                            padding=5, bias=False)
        self.b2 = nn.BatchNorm1d(planes)
        self.c3 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        if downsample:
            self.down_sample = nn.Sequential(
                nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(planes),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualBlock2D(nn.Module):
    def __init__(self, planes, kernel_size=(11, 5), padding=(5, 2), downsample=True):
        super(ResidualBlock2D, self).__init__()
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes * 2, kernel_size=kernel_size, stride=1,
                            padding=padding, bias=False)
        self.b2 = nn.BatchNorm2d(planes * 2)
        self.c3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.down_sample = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.down_sample(x)
        # print(out.shape, identity.shape)
        out += identity
        out = self.relu(out)

        return out
