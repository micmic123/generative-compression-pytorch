import torch
from torch import nn


# Image gradient
class Scharr(nn.Module):
    def __init__(self):
        super(Scharr, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        kernel_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]).float().reshape_as(self.conv1.weight)
        kernel_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]).float().reshape_as(self.conv2.weight)
        self.conv1.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv2.weight = nn.Parameter(kernel_y, requires_grad=False)
        self.rgb2gray_weights = nn.Parameter(torch.tensor([0.0721, 0.7154, 0.2125]).view((-1, 3, 1, 1)),
                                             requires_grad=False)

    def rgb2gray(self, x):
        # x: (B, 3, H, W) RGB image, [-1, 1]
        # x = (x+1.) / 2

        # x = (127.5*(x + 1.0))  # [0, 255]
        # grayValue = 0.0721 * x[:, 0, :, :] + 0.7154 * x[:, 1, :, :] + 0.2125 * x[:, 2, :, :]
        #return grayValue
        return torch.sum(x * self.rgb2gray_weights, dim=1).unsqueeze(1)

    def forward(self, x):
        # x: (B, 3, H, W) RGB image, [-1, 1]
        x = self.rgb2gray(x)  # (B, 1, H, W)
        Ix = self.conv1(x)  # (B, 1, H, W)
        Iy = self.conv2(x)  # (B, 1, H, W)

        return torch.sqrt(Ix**2 + Iy**2)
