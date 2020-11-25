import torch
import torch.nn.functional as F
from torch import nn


#  https://github.com/fab-jul/L3C-PyTorch/blob/master/src/modules/quantizer.py
SIGMA_HARD = 1e7
def to_sym(x, x_min, x_max, L):
    sym_range = x_max - x_min
    bin_size = sym_range / (L-1)
    return x.clamp(x_min, x_max).sub(x_min).div(bin_size).round().long()


def to_bn(S, x_min, x_max, L):
    sym_range = x_max - x_min
    bin_size = sym_range / (L-1)
    return S.float().mul(bin_size).add(x_min)


class Quantizer(nn.Module):
    def __init__(self, levels, sigma=1.0):
        super(Quantizer, self).__init__()
        assert levels.dim() == 1, 'Expected 1D levels, got {}'.format(levels)
        self.levels = levels
        self.sigma = sigma  # 2
        self.L = self.levels.size()[0]

    def __repr__(self):
        return '{}(sigma={})'.format(
                self._get_name(), self.sigma)

    def forward(self, x):
        """
        :param x: NCHW
        :return:
            - x_soft: quantized x
            - x_hard: no_grad of x_soft
            - symbols_hard: indices of x_soft for levels
        """
        assert x.dim() == 4, 'Expected NCHW, got {}'.format(x.size())
        N, C, H, W = x.shape
        # make x into NCm1, where m=H*W
        x = x.view(N, C, H*W, 1)
        # NCmL, d[..., l] gives distance to l-th level
        d = torch.pow(x - self.levels, 2)
        # NCmL, \sum_l d[..., l] sums to 1
        phi_soft = F.softmax(-self.sigma * d, dim=-1)

        # - Calcualte soft assignements ---
        # NCm, soft assign x to levels
        x_soft = torch.sum(self.levels * phi_soft, dim=-1)
        # NCHW
        x_soft = x_soft.view(N, C, H, W)

        # - Calcualte hard assignements ---
        # NCm, symbols_hard[..., i] contains index of symbol to use
        _, symbols_hard = torch.min(d.detach(), dim=-1)
        # NCHW
        symbols_hard = symbols_hard.view(N, C, H, W)
        # NCHW, contains value of symbol to use
        x_hard = self.levels[symbols_hard]

        x_soft.data = x_hard  # assign data, keep gradient

        # , x_hard, symbols_hard
        return x_soft