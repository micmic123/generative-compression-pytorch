from torch import nn
from .blocks import ResBlock, ResBlocks, Conv2dBlock, UpConv2dBlock


class Controller(nn.Module):
    def __init__(self, config, q):
        super(Controller, self).__init__()
        self.config = config
        self.C_level = sorted(config['C_level'])[::-1]  # descending order
        for i in range(len(self.C_level)-1):
            down = [
                nn.InstanceNorm2d(self.C_level[i]),
                Conv2dBlock(self.C_level[i], self.C_level[i+1], 3, 1, 1, norm='none', pad_type='reflect', activation='none')
            ]
            up = [
                nn.InstanceNorm2d(self.C_level[i+1]),
                Conv2dBlock(self.C_level[i+1], self.C_level[i], 3, 1, 1, norm='none', pad_type='reflect', activation='none')
            ]

            setattr(self, f'down_{i}', nn.Sequential(*down))
            setattr(self, f'up_{i}', nn.Sequential(*up))
        self.q = q

    def down(self, z):
        """
        :param
            z: (B, C_level[-1], H', W')
        """
        out = []
        x = z
        for i in range(len(self.C_level)-1):
            down = getattr(self, f'down_{i}')
            x = down(x)
            out.append(x)

        return out[::-1]  # e.g. [(B, 8, 8, 16), (B, 32, 16, 16)]

    def up(self, zs_quantized):
        """
        :param
            zs: ascending order e.g. [(B, 8, 8, 16), (B, 32, 16, 16)]
        """
        zs = zs_quantized[::-1]  # [(B, 32, 16, 16), (B, 8, 8, 16)]
        out = []
        for i in range(len(self.C_level)-1):
            x = zs[i]
            for j in range(i, -1, -1):
                up = getattr(self, f'up_{j}')
                x = up(x)
                x = self.q(x)
            out.append(x)

        return out[::-1]  # ascending order for bpp
