import os
import copy
import math
import gzip
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
from .controller import Controller
from .model import Model
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


class Trainer(nn.Module):
    def __init__(self, config, multigpus=False):
        super(Trainer, self).__init__()
        self.config = config
        self.model = Model(config)
        self.multigpus = multigpus
        self.is_mask = config['mask'] == 1
        self.C_level = config['C_level']
        self.has_controller = config['controller'] == 1
        assert not (self.is_mask and self.has_controller)

        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())
        dis_params = list(self.model.dis.parameters())
        self.encoder_opt = torch.optim.Adam(
            [p for p in encoder_params if p.requires_grad],
            lr=config['lr_encoder'], weight_decay=config['weight_decay'])
        self.decoder_opt = torch.optim.Adam(
            [p for p in decoder_params if p.requires_grad],
            lr=config['lr_decoder'], weight_decay=config['weight_decay'])
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=config['lr_dis'], weight_decay=config['weight_decay'])
        self.model.encoder_test = copy.deepcopy(self.model.encoder)
        self.model.decoder_test = copy.deepcopy(self.model.decoder)
        if self.has_controller:
            controller_params = list(self.model.controller.parameters())
            self.controller_opt = torch.optim.Adam(
                [p for p in controller_params if p.requires_grad],
                lr=config['lr_controller'], weight_decay=config['weight_decay'])
            self.model.controller_test = copy.deepcopy(self.model.controller)

        self.apply(weights_init(config['init']))
        self.itr = 0

    def G_update(self, x):
        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()
        if self.has_controller:
            self.controller_opt.zero_grad()

        loss_recon, loss_fm, loss_G_adv, loss_vgg, loss_grad, loss_G = self.model(x, 'G_update')  # , mask_size
        self.loss_recon = torch.mean(loss_recon).detach()
        self.loss_fm = torch.mean(loss_fm).detach()
        self.loss_G_adv = torch.mean(loss_G_adv).detach()
        self.loss_vgg = torch.mean(loss_vgg).detach()
        self.loss_grad = torch.mean(loss_grad).detach()
        self.loss_G = torch.mean(loss_G).detach()
        self.encoder_opt.step()
        self.decoder_opt.step()

        model = self.model.module if self.multigpus else self.model
        update_average(model.encoder_test, model.encoder)
        update_average(model.decoder_test, model.decoder)
        if self.has_controller:
            self.controller_opt.step()
            update_average(model.controller_test, model.controller)

        return self.loss_recon.item(), self.loss_fm.item(), self.loss_G_adv.item(), self.loss_vgg, self.loss_grad, \
               self.loss_G.item() #, mask_size

    def D_update(self, x):
        self.dis_opt.zero_grad()
        loss_D_real, loss_D_fake, loss_D = self.model(x, 'D_update')
        self.loss_D_real = torch.mean(loss_D_real).detach()
        self.loss_D_fake = torch.mean(loss_D_fake).detach()
        self.loss_D = torch.mean(loss_D).detach()
        self.dis_opt.step()

        return self.loss_D_real.item(), self.loss_D_fake.item(), self.loss_D.item()

    def forward(self, x, mode):
        print('Forward function not implemented.')
        pass

    def test(self, x):
        model = self.model.module if self.multigpus else self.model
        x, x_recon, x_recon_ema = model.test(x)
        return x, x_recon, x_recon_ema

    def save(self, snapshot_dir, filename):
        model = self.model.module if self.multigpus else self.model

        snapshot = {
            'itr': self.itr,
            'config': self.config,
            'encoder': model.encoder.state_dict(),
            'encoder_test': model.encoder_test.state_dict(),
            'decoder': model.decoder.state_dict(),
            'decoder_test': model.decoder_test.state_dict(),
            'controller': model.controller.state_dict(),
            'controller_test': model.controller_test.state_dict(),
            'dis': model.dis.state_dict(),
            'encoder_opt': self.encoder_opt.state_dict(),
            'decoder_opt': self.decoder_opt.state_dict(),
            'dis_opt': self.dis_opt.state_dict(),
            'controller_opt': self.controller_opt.state_dict()
        }

        torch.save(snapshot, os.path.join(snapshot_dir, filename))

    def load(self, path):
        model = self.model.module if self.multigpus else self.model

        snapshot = torch.load(path)
        self.itr = snapshot['itr']
        model.encoder.load_state_dict(snapshot['encoder'])
        model.encoder_test.load_state_dict(snapshot['encoder_test'])
        model.decoder.load_state_dict(snapshot['decoder'])
        model.decoder_test.load_state_dict(snapshot['decoder_test'])
        model.dis.load_state_dict(snapshot['dis'])
        self.encoder_opt.load_state_dict(snapshot['encoder_opt'])
        self.decoder_opt.load_state_dict(snapshot['decoder_opt'])
        self.dis_opt.load_state_dict(snapshot['dis_opt'])
        if 'controller' in snapshot:
            model.controller.load_state_dict(snapshot['controller'])
            model.controller_test.load_state_dict(snapshot['controller_test'])
            self.controller_opt.load_state_dict(snapshot['controller_opt'])

        print(f'Loaded from itr: {self.itr}.')

    def encode(self, x):
        model = self.model.module if self.multigpus else self.model
        return model.encode(x)

    def encode_ema(self, x):
        model = self.model.module if self.multigpus else self.model
        return model.encode_ema(x)

    def decode(self, z_compressed, shape=(1, 8, 16, 16), cuda=True):
        model = self.model.module if self.multigpus else self.model
        return model.decode(z_compressed, shape, cuda)

    def decode_ema(self, z_compressed, shape=(1, 8, 16, 16), cuda=True):
        model = self.model.module if self.multigpus else self.model
        return model.decode_ema(z_compressed, shape, cuda)
