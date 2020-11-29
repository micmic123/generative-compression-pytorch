import math
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim


def MSE(x, y):
    # x, y: 4D [0, 1]
    return torch.mean((x-y)**2, dim=[1,2,3])


def PSNR(x, y, range=(-1,1)):
    # x, y: 4D
    if range == (-1, 1):
        x = (x+1)/2
        y = (y+1)/2

    mse = MSE(x, y)
    return 10*torch.log10(1. / mse)  # (N,)


def MS_SSIM(x, y, range=(-1, 1)):
    if range == (-1, 1):
        x = (x+1)/2
        y = (y+1)/2

    return ms_ssim(x, y, data_range=1, size_average=False)  # (N,)
