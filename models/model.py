import os
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn, optim
# from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
from .blocks import ResBlock, ResBlocks, Conv2dBlock, UpConv2dBlock, LinearBlock
from .quantizer import Quantizer
from .discriminator import Discriminator
from .vgg import VGG16


# EMA
def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.dis = Discriminator(config)
        self.vgg = VGG16()
        self.vgg_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.criterion = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()

        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())
        dis_params = list(self.dis.parameters())
        self.encoder_opt = torch.optim.Adam(
            [p for p in encoder_params if p.requires_grad],
            lr=config['lr_encoder'], weight_decay=config['weight_decay'])
        self.decoder_opt = torch.optim.Adam(
            [p for p in decoder_params if p.requires_grad],
            lr=config['lr_decoder'], weight_decay=config['weight_decay'])
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=config['lr_dis'], weight_decay=config['weight_decay'])

        self.encoder_test = copy.deepcopy(self.encoder)
        self.decoder_test = copy.deepcopy(self.decoder)
        self.apply(weights_init(config['init']))

    def get_scheduler(self):
        return self.encoder_lr_sche, self.decoder_lr_sche, self.dis_lr_sche

    def D_out_decompose(self, D_out):
        # D_out: (num_D, n_layers+2, (B, ., ., .) featmap)
        score = []
        feat = []
        for x in D_out:
            score.append(x[-1])
            for y in x[:-1]:
                feat.append(y)

        return score, feat

    def G_update(self, x):
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

        z_quantized = self.encoder(x)
        x_recon = self.decoder(z_quantized)
        score_recon, feat_recon = self.D_out_decompose(self.D(x_recon))
        score_x, feat_x = self.D_out_decompose(self.D(x))

        # recon loss
        self.loss_recon = self.criterion(x_recon, x)

        # feature matching loss
        self.loss_fm = torch.mean(torch.stack([self.criterion(a, b) for a, b in zip(feat_recon, feat_x)]))

        # VGG perceptual loss
        x_vgg, x_recon_vgg = self.vgg(x), self.vgg(x_recon)
        vgg_loss = 0
        for i, (f1, f2) in enumerate(zip(x_vgg, x_recon_vgg)):
            vgg_loss += self.vgg_weights[i] * self.criterion_L1(f1.detach(), f2)
        self.vgg_loss = vgg_loss


        # adversarial loss
        self.loss_G_adv = torch.mean(torch.stack([F.mse_loss(score, torch.ones_like(score)) for score in score_recon]))

        self.loss_G = self.config['recon_w']*self.loss_recon + self.config['fm_w']*self.loss_fm + \
                      self.config['adv_w']*self.loss_G_adv + self.config['vgg_w']*self.vgg_loss

        self.loss_G.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()
        update_average(self.encoder_test, self.encoder)
        update_average(self.decoder_test, self.decoder)

        return self.loss_recon.item(), self.loss_fm.item(), self.loss_G_adv.item(), self.loss_G.item()

    def D_update(self, x):
        self.dis_opt.zero_grad()

        with torch.no_grad():
            z_quantized = self.encoder(x)
            x_recon = self.decoder(z_quantized)
        score_recon, feat_recon = self.D_out_decompose(self.D(x_recon))
        score_x, feat_x = self.D_out_decompose(self.D(x))

        # adversarial loss
        self.loss_D_real = torch.mean(torch.stack([self.criterion(score, torch.ones_like(score)) for score in score_x]))
        self.loss_D_fake = torch.mean(torch.stack([self.criterion(score, torch.zeros_like(score)) for score in score_recon]))
        self.loss_D = self.config['adv_w']*torch.mean(self.loss_D_real + self.loss_D_fake)

        self.loss_D.backward()
        self.dis_opt.step()

        return self.loss_D_real.item(), self.loss_D_fake.item(), self.loss_D.item()

    def forward(self, x, mode):
        print('Forward function not implemented.')
        pass

    def test(self, x):
        self.eval()
        with torch.no_grad():
            z_quantized = self.encoder(x)
            x_recon = self.decoder(z_quantized)

            z_quantized_ema = self.encoder_test(x)
            x_recon2 = self.decoder_test(z_quantized_ema)
        self.train()

        return x_recon, x_recon2

    def save(self, snapshot_dir, itr):
        snapshot = {
            'itr': itr,
            'config': self.config,
            'encoder': self.encoder,
            'decoder': self.decoder.state_dict(),
            'dis': self.dis.state_dict(),
            'encoder_opt': self.encoder_opt.state_dict(),
            'decoder_opt': self.decoder_opt.state_dict(),
            'dis_opt': self.dis_opt.state_dict()
        }

        torch.save(snapshot, os.path.join(snapshot_dir, f'itr_{itr:07}.pt'))

    def load(self, path):
        snapshot = torch.load(path)
        self.encoder.load_state_dict(snapshot['encoder'])
        self.decoder.load_state_dict(snapshot['decoder'])
        self.dis.load_state_dict(snapshot['dis'])
        self.encoder_opt.load_state_dict(snapshot['encoder_opt'])
        self.decoder_opt.load_state_dict(snapshot['decoder_opt'])
        self.dis_opt.load_state_dict(snapshot['dis_opt'])

        print(f'Loaded from itr: {snapshot["itr"]}.')

        return snapshot['itr']


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        df = config['downscale_factor']
        image_shape = config['image_shape']
        up_channel = config['enc_up_channel']
        self.C = config['C']
        self.L = config['L']

        self.model = [
            Conv2dBlock(image_shape[0], up_channel, 7, 1, 3, norm='in', pad_type='reflect'),
        ]
        for i in range(df):
            self.model.append(Conv2dBlock(up_channel, 2*up_channel, 4, 2, 1, norm='in', pad_type='reflect'))
            up_channel *= 2
        self.model.append(Conv2dBlock(up_channel, self.C, 3, 1, 1, norm='in', pad_type='reflect'))
        self.model = nn.Sequential(*self.model)

        self.levels = nn.Parameter(torch.linspace(-(self.L // 2), (self.L // 2), self.L), requires_grad=False)
        self.q = Quantizer(self.levels, config['Q_std'])

    def forward(self, x):
        '''
        :param
            x: Tensor for image, (B, 3, H, W)
        :return z_quantized, (B, C, H', W')
        '''
        z = self.model(x)
        z_quantized, _, _ = self.q(z)

        return z_quantized

    def get_z(self, x):
        z = self.model(x)

        return z


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        image_shape = config['image_shape']
        C = config['C']
        df = config['downscale_factor']
        up_channel = config['dec_up_channel']
        res_num = config['dec_res_num']
        self.decoder = [
            Conv2dBlock(C, up_channel, 3, 1, 1, norm='in', pad_type='reflect'),
            ResBlocks(res_num, up_channel, norm='in', activation='relu', pad_type='reflect')
        ]
        for i in range(df):
            self.decoder.append(UpConv2dBlock(up_channel // (2 ** i)))
        self.decoder.append(Conv2dBlock(up_channel // (2 ** df), image_shape[0], 7, 1, padding=3, norm='none',
                                        activation='tanh', pad_type='reflect'))
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, z_quantized):
        """
        :param z_quantized: (B, C, H', W')
        """
        out = self.decoder(z_quantized)

        return out


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun