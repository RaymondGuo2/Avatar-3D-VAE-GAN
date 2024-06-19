import torch
import torch.nn as nn
from utils import get_conv_output_size


# Source literature: kernel size {11,5,5,5,8}, stride {4,2,2,2,1}
class Encoder(nn.Module):
    def __init__(self, img_size=28):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=False)
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=False)
        )
        self.conv4 = nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=False)
        )
        self.conv5 = nn.Sequential(
            torch.nn.Conv2d(512, 400, kernel_size=2, stride=1),
            torch.nn.BatchNorm2d(400),
            torch.nn.ReLU(inplace=False)
        )
        self.flatten = nn.Flatten()
        # conv_output_size = get_conv_output_size(img_size)
        # Usually substitute 400 with conv_output_size
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(400, 200)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        z_mean = self.fc1(x)
        z_var = self.fc2(x)
        return z_mean, z_var


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            torch.nn.ConvTranspose3d(200, 512, kernel_size=4, stride=2, padding=0),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=False)
        )
        self.conv3 = nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU(inplace=False)
        )
        self.conv4 = nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(inplace=False)
        )
        self.conv5 = nn.Sequential(
            torch.nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, 200, 1, 1, 1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2, inplace=False)
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2, inplace=False)
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2, inplace=False)
        )
        self.conv4 = nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(0.2, inplace=False)
        )
        self.conv5 = nn.Sequential(
            torch.nn.ConvTranspose3d(512, 1, kernel_size=4, stride=2, padding=0),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = x.view(-1, 1, 64, 64, 64)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


