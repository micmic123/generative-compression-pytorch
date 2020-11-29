import os
import yaml
import time
from glob import glob
from shutil import copy2
import torch
from torchvision import utils as vutils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from dataset import inference_transform
from metrics import PSNR, MS_SSIM


def get_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        config = compatible(config)
        return config


def init(args):
    base_dir = f'./results/{args.name}'
    snapshot_dir = os.path.join(base_dir, 'snapshots')
    output_dir = os.path.join(base_dir, 'outputs')
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    config = get_config(args.config)
    copy2(args.config, os.path.join(base_dir, 'config.yaml'))
    summary_writers = get_summary_writers(config, log_dir)

    return base_dir, snapshot_dir, output_dir, summary_writers, config


def compatible(config):
    if config['mask'] == 1 or config['controller'] == 1:
        config['C'] = config['C_level'][-1]
    if 'C_w' not in config:
        config['C_w'] = [0.25, 0.25, 0.25, 0.25]
    if 'eval_itr' not in config:
        config['eval_itr'] = 1000
    if 'controller_v' not in config:
        config['controller_v'] = 2
    if 'controller_q' not in config:
        config['controller_q'] = 0
    if 'match_w' not in config:
        config['match_w'] = 0
    return config


def get_eval_list(eval_path='./eval_samples/'):
    paths = glob(os.path.join(eval_path, '*.png'))
    paths += glob(os.path.join(eval_path, '*.jpg'))
    assert paths
    imgs = []
    for path in paths:
        imgs.append(inference_transform(Image.open(path).convert('RGB')).unsqueeze(0))

    return imgs


def eval_model(trainer, imgs):
    # imgs: list of tensor image
    psnrs, ms_ssims = [], []
    if trainer.has_controller:
        for x in imgs:
            x, x_recon, x_recon_ema = trainer.test(x, verbose=False)
            psnrs.append(PSNR(x, x_recon))
            ms_ssims.append(MS_SSIM(x, x_recon_ema))

        psnrs = torch.stack(psnrs, dim=0).mean(dim=0).cpu()
        ms_ssims = torch.stack(ms_ssims, dim=0).mean(dim=0).cpu()
        trainer.level_log['PSNR'] = {i: torch.mean(psnrs[i].detach()) for i in range(len(trainer.C_level))}
        trainer.level_log['MS_SSIM'] = {i: torch.mean(ms_ssims[i].detach()) for i in range(len(trainer.C_level))}
        psnr = [f'{p:.4f}' for p in psnrs]
        ms_ssim = [f'{p:.4f}' for p in ms_ssims]
        print('eval: ', psnr, ms_ssim)
    else:
        for x in imgs:
            x = x.cuda()
            z, z_shape = trainer.encode(x)
            x_recon = trainer.decode(z, shape=z_shape)
            z_ema, z_ema_shape = trainer.encode_ema(x)
            x_recon_ema = trainer.decode_ema(z_ema, shape=z_ema_shape)

            psnrs.append(PSNR(x, x_recon))
            ms_ssims.append(MS_SSIM(x, x_recon_ema))

        psnrs = torch.cat(psnrs, dim=0).mean(dim=0).cpu().detach()
        ms_ssims = torch.cat(ms_ssims, dim=0).mean(dim=0).cpu().detach()
        trainer.eval_psnr = psnrs
        trainer.eval_ms_ssim = ms_ssims
        print('eval: ', f'{psnrs.item():.4f}', f'{ms_ssims.item():.4f}')


def get_summary_writers(config, log_dir):
    if config['controller'] == 1:
        writers = {
            'total': SummaryWriter(os.path.join(log_dir, 'total'))
        }
        for i in range(len(config['C_level'])):
            writers[i] = SummaryWriter(os.path.join(log_dir, f'level_{i}'))

        return writers
    else:
        return SummaryWriter(log_dir)


def write_loss(itr, model, writers, eval=False):
    if model.has_controller:
        for k, writer in writers.items():
            if k == 'total': continue
            if eval:
                writer.add_scalar('PSNR', model.level_log['PSNR'][k], itr)
                writer.add_scalar('MS_SSIM', model.level_log['MS_SSIM'][k], itr)
            else:
                writer.add_scalar('loss_recon_level', model.level_log['loss_recon_level'][k], itr)
                writer.add_scalar('loss_fm_level', model.level_log['loss_fm_level'][k], itr)
                writer.add_scalar('loss_vgg_level', model.level_log['loss_vgg_level'][k], itr)
                writer.add_scalar('loss_G_adv_level', model.level_log['loss_G_adv_level'][k], itr)
                writer.add_scalar('loss_D_fake', model.level_log['loss_D_fake'][k], itr)
                if k < len(model.C_level)-1:
                    writer.add_scalar('loss_match_level', model.level_log['loss_match_level'][k], itr)
        writer = writers['total']
    else:
        writer = writers
        if eval:
            writer.add_scalar('PSNR', model.eval_psnr, itr)
            writer.add_scalar('MS_SSIM', model.eval_ms_ssim, itr)
            return
    members = [attr for attr in dir(model)
               if ((not callable(getattr(model, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    for m in members:
        writer.add_scalar(m, getattr(model, m), itr)


def denormalize(img):
    return (127.5*(img.float() + 1.0)).permute((1,2,0)).numpy().astype(np.uint8)


def save_image(tensor, path):
    tensor = tensor.cpu()
    img = denormalize(tensor)
    Image.fromarray(img).save(path)


def save_grid(tensor, path, nrow=4):
    grid = vutils.make_grid(tensor.cpu(), nrow=nrow)
    img = denormalize(grid)
    Image.fromarray(img).save(path)


def lr_schedule(optimizer, lr_original, beta=0.1):
    lr = lr_original * beta
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        if old_lr != lr:
            param_group['lr'] = lr


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))