import os
import copy
import math
import gzip as compressor
from random import randint
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
# from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
from .blocks import ResBlock, ResBlocks, Conv2dBlock, UpConv2dBlock, LinearBlock
from .quantizer import Quantizer
from .discriminator import Discriminator
from .controller import Controller, Controller2
from .vgg import VGG16
from .loss import Scharr


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
        self.is_mask = config['mask'] == 1
        self.C_level = config['C_level']
        self.C_w = config['C_w']
        self.L = config['L']
        self.has_controller = config['controller'] == 1
        self.levels = nn.Parameter(torch.linspace(-(self.L // 2), (self.L // 2), self.L), requires_grad=False)
        self.controller_v = config['controller_v']
        self.C_match_w = torch.tensor(self.C_w[:-1]).float()
        self.C_match_w = self.C_match_w / self.C_match_w.sum()
        assert self.controller_v == 1 or self.controller_v == 2

        self.quantizer = Quantizer(self.levels, config['Q_std'])
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.dis = Discriminator(config)
        self.vgg = VGG16()
        if self.has_controller:
            self.criterion = nn.MSELoss(reduction='none')
            if self.controller_v == 1:
                self.controller = Controller(config, self.quantizer)
            elif self.controller_v == 2:
                self.controller = Controller2(config, self.quantizer)
        else:
            self.criterion = nn.MSELoss()
        self.vgg_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.scharr = Scharr()
        self.criterion_L1 = nn.L1Loss()

        self.is_mask = config['mask'] == 1
        self.C_level = config['C_level']
        self.has_controller = config['controller'] == 1

    def D_out_decompose(self, D_out):
        # D_out: (num_D, n_layers+2, (B, ., ., .) featmap)
        # D_out[0]: [(B, ., ., .), (B, ., ., .), ..., (B, ., ., .)]
        score = []  # num_D list of (B, ., ., .)
        feat = []  # num_D*(n_layers+1) list of (B, ., ., .)
        for x in D_out:
            score.append(x[-1])
            for y in x[:-1]:
                feat.append(y)

        return score, feat

    def levelwise_loss(self, loss_tmp, name, batchsize):
        # loss_tmp: (B*(len(C_Level)), )
        loss = 0
        loss_ls = []
        for i, l_level in enumerate(loss_tmp.split(batchsize)):
            # l_level: (B, )
            loss_level = torch.mean(l_level)
            loss += self.C_w[i] * loss_level
            # setattr(self, f'loss_{name}_level{i}', loss_level)
            loss_ls.append(loss_level.detach())
        return loss, loss_ls

    def G_update(self, x):
        batchsize = x.shape[0]
        x = x.cuda()
        z = self.encoder(x)
        z_quantized = self.quantizer(z)
        mask_size = self.config['C']
        if self.is_mask:
            mask_size = self.C_level[randint(0, len(self.C_level)-1)]
            mask = torch.zeros_like(z_quantized, requires_grad=False).cuda()
            mask[:, :mask_size] = 1.
            z_quantized = z_quantized * mask
        elif self.has_controller:
            if self.controller_v == 1:
                zs = self.controller.down(z)  # [(B, 8, 8, 16), (B, 32, 16, 16)]
                zs_quantized = [self.quantizer(z_) for z_ in zs]
            elif self.controller_v == 2:
                zs_quantized = self.controller.down(z_quantized)  # [(B, 8, 8, 16), (B, 32, 16, 16)]
            z_q_all = self.controller.up(zs_quantized)  # [(B, 128, 16, 16) * (len(C_level)-1)]
            z_q_all.append(z_quantized)  # [(B, 128, 16, 16) * len(C_level)]
            z_quantized = torch.cat(z_q_all, dim=0)  # (B*(len(C_Level)), 128, 16, 16)
            x = x.repeat((len(self.C_level), 1, 1, 1))

        x_recon = self.decoder(z_quantized)
        score_recon, feat_recon = self.D_out_decompose(self.dis(x_recon))
        score_x, feat_x = self.D_out_decompose(self.dis(x))

        # controller match loss
        if self.has_controller and self.controller_v == 2:
            z_pred = torch.stack(z_q_all[:-1])  # (len(C_level)-1, B, 128, 16, 16)
            z_gt = z_q_all[-1].unsqueeze(0).expand_as(z_pred)
            loss_match_ls = torch.mean(self.criterion(z_pred, z_gt), dim=[1,2,3,4])  # (len(C_level)-1, )
            loss_match = torch.sum(self.C_match_w.cuda() * loss_match_ls)
        else:
            loss_match = 0
            loss_match_ls = None

        # recon loss
        if self.has_controller:
            loss_recon_tmp = torch.mean(self.criterion(x_recon, x), dim=[1, 2, 3])  # (B*(len(C_Level)), )
            loss_recon, loss_recon_ls = self.levelwise_loss(loss_recon_tmp, 'recon', batchsize)
        else:
            loss_recon = self.criterion(x_recon, x)

        # feature matching loss
        if self.has_controller:
            loss_fm_tmp = torch.mean(torch.stack(
                [torch.mean(self.criterion(a, b), dim=[1, 2, 3]) for a, b in zip(feat_recon, feat_x)]), dim=0)
            loss_fm, loss_fm_ls = self.levelwise_loss(loss_fm_tmp, 'fm', batchsize)
        else:
            loss_fm = torch.mean(torch.stack([self.criterion(a, b) for a, b in zip(feat_recon, feat_x)]))

        # VGG perceptual loss
        x_vgg, x_recon_vgg = self.vgg(x), self.vgg(x_recon)
        if self.has_controller:
            loss_vgg_tmp = torch.sum(torch.stack(
                [torch.mean(self.vgg_weights[i] * self.criterion(f1.detach(), f2), dim=[1, 2, 3])
                 for i, (f1, f2) in enumerate(zip(x_vgg, x_recon_vgg))]), dim=0)
            loss_vgg, loss_vgg_ls = self.levelwise_loss(loss_vgg_tmp, 'vgg', batchsize)
        else:
            loss_vgg = 0
            for i, (f1, f2) in enumerate(zip(x_vgg, x_recon_vgg)):
                loss_vgg += self.vgg_weights[i] * self.criterion(f1.detach(), f2)
            loss_vgg = loss_vgg

        # image gradient loss
        x_grad = self.scharr(x)
        x_recon_grad = self.scharr(x_recon)
        loss_grad = self.criterion(x_grad, x_recon_grad).mean()

        # adversarial loss
        if self.has_controller:
            loss_G_adv_tmp = torch.mean(torch.stack(
                [torch.mean(self.criterion(score, torch.ones_like(score)), dim=[1, 2, 3])
                 for score in score_recon]), dim=0)
            loss_G_adv, loss_G_adv_ls = self.levelwise_loss(loss_G_adv_tmp, 'G_adv', batchsize)
        else:
            loss_G_adv = torch.mean(torch.stack([self.criterion(score, torch.ones_like(score)) for score in score_recon]))
            loss_recon_ls, loss_fm_ls, loss_G_adv_ls, loss_vgg_ls = None, None, None, None

        loss_G = self.config['recon_w']*loss_recon + self.config['fm_w']*loss_fm + \
                 self.config['adv_w']*loss_G_adv + self.config['vgg_w']*loss_vgg + \
                 self.config['grad_w']*loss_grad + self.config['match_w']*loss_match

        loss_G.backward()

        return loss_recon, loss_fm, loss_G_adv, loss_vgg, loss_grad, loss_match, \
               loss_G, loss_recon_ls, loss_fm_ls, loss_G_adv_ls, loss_vgg_ls, loss_match_ls  # , mask_size

    def D_update(self, x):
        batchsize = x.shape[0]
        x = x.cuda()

        with torch.no_grad():
            z = self.encoder(x)
            z_quantized = self.quantizer(z)
            mask_size = self.config['C']
            if self.is_mask:
                mask_size = self.C_level[randint(0, len(self.C_level)-1)]
                mask = torch.zeros_like(z_quantized, requires_grad=False).cuda()
                mask[:, :mask_size] = 1.
                z_quantized = z_quantized * mask
            elif self.has_controller:
                if self.controller_v == 1:
                    zs = self.controller.down(z)  # [(B, 8, 8, 16), (B, 32, 16, 16)]
                    zs_quantized = [self.quantizer(z_) for z_ in zs]
                elif self.controller_v == 2:
                    zs_quantized = self.controller.down(z_quantized)  # [(B, 8, 8, 16), (B, 32, 16, 16)]
                z_q_all = self.controller.up(zs_quantized)
                z_q_all.append(z_quantized)
                z_quantized = torch.cat(z_q_all, dim=0)
            x_recon = self.decoder(z_quantized)
        score_recon, feat_recon = self.D_out_decompose(self.dis(x_recon))
        score_x, feat_x = self.D_out_decompose(self.dis(x))

        # adversarial loss
        if self.has_controller:
            loss_D_real = torch.mean(torch.stack([self.criterion(score, torch.ones_like(score)).mean() for score in score_x]))
            loss_D_fake_tmp = torch.mean(torch.stack(
                [torch.mean(self.criterion(score, torch.zeros_like(score)), dim=[1, 2, 3])
                 for score in score_recon]), dim=0)
            loss_D_fake, loss_D_fake_ls = self.levelwise_loss(loss_D_fake_tmp, 'D_fake', batchsize)
            loss_D = self.config['adv_w'] * torch.mean(loss_D_real + loss_D_fake)
        else:
            loss_D_real = torch.mean(torch.stack([self.criterion(score, torch.ones_like(score)) for score in score_x]))
            loss_D_fake = torch.mean(torch.stack([self.criterion(score, torch.zeros_like(score)) for score in score_recon]))
            loss_D = self.config['adv_w']*torch.mean(loss_D_real + loss_D_fake)
            loss_D_fake_ls = None

        loss_D.backward()

        return loss_D_real, loss_D_fake, loss_D, loss_D_fake_ls

    def forward(self, x, mode):
        if mode == 'G_update':
            return self.G_update(x)
        elif mode == 'D_update':
            return self.D_update(x)

    def test(self, x, filename='test', verbose=True):
        x = x.cuda()
        self.eval()
        with torch.no_grad():
            if self.is_mask or self.has_controller:
                if self.is_mask:
                    x = x[0].unsqueeze(0).expand((len(self.C_level), -1, -1, -1))
                    z = self.encoder(x)
                    z_quantized = self.quantizer(z)
                    z_ema = self.encoder_test(x)
                    z_quantized_ema = self.quantizer(z_ema)
                    masks = []
                    for mask_size in self.C_level:
                        mask = torch.zeros(z_quantized.shape[1:], requires_grad=False)
                        mask[:mask_size] = 1.
                        masks.append(mask)
                    masks = torch.stack(masks).cuda()
                    z_quantized = z_quantized * masks
                    z_quantized_ema = z_quantized_ema * masks

                    # print size
                    print('[ test ]') if verbose else None
                    z_np = z_quantized.cpu().numpy().astype(np.int8)
                    z_ema_np = z_quantized_ema.cpu().numpy().astype(np.int8)
                    for i, level in enumerate(self.C_level):
                        z_comp = compressor.compress(z_np[i][:level])
                        z_comp2 = compressor.compress(z_np[i])
                        z_ema_comp = compressor.compress(z_ema_np[i][:level])
                        z_ema_comp2 = compressor.compress(z_ema_np[i])
                        print(f'{level:>3}: {len(z_comp)}({len(z_comp2)}) bytes, '
                              f'{len(z_ema_comp)}({len(z_ema_comp2)}) bytes')  if verbose else None
                else:
                    x = x[0].unsqueeze(0)
                    z = self.encoder(x)
                    z_q = self.quantizer(z)
                    if self.controller_v == 1:
                        zs = self.controller.down(z)  # [(B, 8, 8, 16), (B, 32, 16, 16)]
                        zs_quantized = [self.quantizer(z_) for z_ in zs]
                    elif self.controller_v == 2:
                        zs_quantized = self.controller.down(z_q)  # [(B, 8, 8, 16), (B, 32, 16, 16)]
                    z_q_all = self.controller.up(zs_quantized)  # [(1, 128, 16, 16) * (len(C_level)-1)]
                    z_q_all.append(z_q)  # [(1, 128, 16, 16) * len(C_level)]
                    z_quantized = torch.cat(z_q_all, dim=0)  # (len(C_Level), 128, 16, 16)

                    z_ema = self.encoder_test(x)
                    zs_ema = self.controller_test.down(z_ema)
                    z_q_ema = self.quantizer(z_ema)
                    if self.controller_v == 1:
                        zs_quantized_ema = [self.quantizer(z_) for z_ in zs_ema]
                    elif self.controller_v == 2:
                        zs_quantized_ema = zs_ema
                    z_q_all_ema = self.controller_test.up(zs_quantized_ema)
                    z_q_all_ema.append(z_q_ema)
                    z_quantized_ema = torch.cat(z_q_all_ema, dim=0)

                    # print size
                    print(f'[{filename}]') if verbose else None
                    x_size = x.shape[-1] * x.shape[-2]
                    z_all = zs_quantized + [z_q]
                    z_np = [z_.cpu().numpy().astype(np.int8) for z_ in z_all]
                    z_ema_all = zs_quantized_ema + [z_q_ema]
                    z_ema_np = [z_.cpu().numpy().astype(np.int8) for z_ in z_ema_all]
                    for i, level in enumerate(self.C_level):
                        z_comp = compressor.compress(z_np[i])
                        z_ema_comp = compressor.compress(z_ema_np[i])
                        print(f'{level:>3}: '
                              f'{len(z_comp):6}B ({len(z_comp)*8/(x_size):.4f}bpp), '
                              f'{len(z_ema_comp):6}B ({len(z_ema_comp)*8/(x_size):.4f}bpp) ') if verbose else None
                    x = x.expand((len(self.C_level), -1, -1, -1))

            else:
                z_quantized = self.quantizer(self.encoder(x))
                z_quantized_ema = self.quantizer(self.encoder_test(x))
            x_recon = self.decoder(z_quantized)
            x_recon_ema = self.decoder_test(z_quantized_ema)
        self.train()

        return x, x_recon, x_recon_ema

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            z_quantized = self.quantizer(self.encoder(x))
        z_np = z_quantized.cpu().numpy().astype(np.int8)
        z_compressed = compressor.compress(z_np)
        self.train()

        return z_compressed, z_np.shape

    def encode_ema(self, x):
        self.eval()
        with torch.no_grad():
            z_quantized = self.quantizer(self.encoder_test(x))
        z_np = z_quantized.cpu().numpy().astype(np.int8)
        z_compressed = compressor.compress(z_np)
        self.train()

        return z_compressed, z_np.shape

    def decode(self, z_compressed, shape=(1, 8, 16, 16), cuda=True):
        self.eval()
        z_bytes = compressor.decompress(z_compressed)
        z_np = np.frombuffer(z_bytes, dtype=np.int8).reshape(shape).copy()
        z_quantized = torch.from_numpy(z_np).type(torch.float32)
        if cuda:
            z_quantized = z_quantized.cuda()
        with torch.no_grad():
            x_recon = self.decoder(z_quantized)
        self.train()

        return x_recon

    def decode_ema(self, z_compressed, shape=(1, 8, 16, 16), cuda=True):
        self.eval()
        z_bytes = compressor.decompress(z_compressed)
        z_np = np.frombuffer(z_bytes, dtype=np.int8).reshape(shape).copy()
        z_quantized = torch.from_numpy(z_np).type(torch.float32)
        if cuda:
            z_quantized = z_quantized.cuda()
        with torch.no_grad():
            x_recon = self.decoder_test(z_quantized)
        self.train()

        return x_recon


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        df = config['downscale_factor']
        image_shape = config['image_shape']
        up_channel = config['enc_up_channel']
        self.C = config['C']

        self.model = [
            Conv2dBlock(image_shape[0], up_channel, 7, 1, 3, norm='in', pad_type='reflect'),
        ]
        for i in range(df):
            self.model.append(Conv2dBlock(up_channel, 2*up_channel, 4, 2, 1, norm='in', pad_type='reflect'))
            up_channel *= 2
        self.model.append(Conv2dBlock(up_channel, self.C, 3, 1, 1, norm='none', pad_type='reflect', activation='none')) # important
        self.model = nn.Sequential(*self.model)


    def forward(self, x):
        '''
        :param
            x: Tensor for image, (B, 3, H, W)
        :return z, (B, C, H', W')
        '''
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
            nn.InstanceNorm2d(C),
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