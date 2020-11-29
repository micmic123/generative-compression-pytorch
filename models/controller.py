from torch import nn
from .blocks import ResBlock, ResBlocks, Conv2dBlock, ChannelResBlock


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


class Controller2(nn.Module):
    def __init__(self, config, q):
        super(Controller2, self).__init__()
        self.config = config
        self.C_level = sorted(config['C_level'])  # ascending order
        self.controller_q = config['controller_q'] == 1
        self.q = q
        for i in range(len(self.C_level)-1):
            in_dim = self.C_level[i]
            model = [
                # nn.InstanceNorm2d(in_dim),
                ChannelResBlock(in_dim, self.C_level[-1], norm='in', activation='relu', pad_type='reflect')
            ]
            setattr(self, f'model_{i}', nn.Sequential(*model))

    def forward(self, z_quantized):
        """
        :param
            z: (B, self.C_level[-1], H', W')
        """
        out = []
        for i in range(len(self.C_level)-1):
            x = z_quantized[:, :self.C_level[i]]
            model = getattr(self, f'model_{i}')
            x = model(x)
            if self.controller_q:
                x = self.q(x)
            out.append(x)

        return out

    def forward_level(self, z, level):
        """
        :param
            z: (B, self.C_level[level], H', W')
        """
        model = getattr(self, f'model_{level}')
        return model(z)

    def down(self, z_quantized):
        """
        :param
            z_quantized: (B, C_level[-1], H', W')
        """
        out = []
        for i in range(len(self.C_level) - 1):
            out.append(z_quantized[:, :self.C_level[i]])
        return out

    def up(self, zs_quantized):
        """
        :param
            zs: ascending order e.g. [(B, 8, 8, 16), (B, 32, 16, 16)]
        """
        out = []
        for i, z in enumerate(zs_quantized):
            z_up = self.forward_level(z, i)
            if self.controller_q:
                z_up = self.q(z_up)
            out.append(z_up)
        return out
