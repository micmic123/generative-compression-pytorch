import os
import argparse
import gzip as compressor
from collections import Counter
from time import time
from glob import glob
from PIL import Image
import torch
import numpy as np
from utils import save_image, get_config, save_grid, get_eval_list, eval_model
from metrics import PSNR, MS_SSIM
from dataset import inference_transform
from models.trainer import Trainer



parser = argparse.ArgumentParser()
# parser.add_argument('--config', help='config file path', type=str)
parser.add_argument('--device', help='CUDA_VISIBLE_DEVICES number', default='3', type=str)
parser.add_argument('--img', help='image path', type=str)
parser.add_argument('--snapshot', help='snapshot path', type=str, required=True)
parser.add_argument('--test', help='snapshot path', action='store_true')
args = parser.parse_args()


base_path = '/'.join(args.snapshot.split('/')[:-2])
args.config = os.path.join(base_path, 'config.yaml')
example_dir = os.path.join(base_path, 'examples')
os.makedirs(example_dir, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
config = get_config(args.config)
if 'mask' not in config:
    config['mask'] = 0
if 'controller' not in config:
    config['controller'] = 0

print('[device]', args.device)
print('[config]', args.config)
msg = f'======================= {args.snapshot} ======================='
print(msg)
for k, v in config.items():
    print(f'  {k}: ', v)
print('='*len(msg))
print()

model = Trainer(config)
model.cuda()
model.load(args.snapshot)

if args.test:
    imgs = get_eval_list()
    eval_model(model, imgs)
    exit()


if args.img:
    paths = [args.img]
else:
    paths = glob('./samples/*.png')
    paths += glob('./samples/*.jpg')
fiename_maxlen = max([len(os.path.basename(path)) for path in paths])


if config['mask']:
    exit()

for path in paths:
    filename = os.path.basename(path).split('.')[0]
    img = Image.open(path).convert('RGB')
    x = inference_transform(img)
    x = x.unsqueeze(0)
    if config['controller']:
        x, x_recon, x_recon_ema = model.test(x, filename=filename)
        out = torch.cat([x.detach(), x_recon.detach(), x_recon_ema.detach()], dim=0)
        save_grid(out, os.path.join(example_dir, f'{filename}_{model.itr:08}_grid.png'), nrow=len(model.C_level))
        psnr = PSNR(x, x_recon)
        psnr_ema = PSNR(x, x_recon_ema)
        ms_ssim = MS_SSIM(x, x_recon)
        ms_ssim_ema = MS_SSIM(x, x_recon_ema)

        psnr = [f'{p:.4f}' for p in psnr]
        psnr_ema = [f'{p:.4f}' for p in psnr_ema]
        ms_ssim = [f'{p:.4f}' for p in ms_ssim]
        ms_ssim_ema = [f'{p:.4f}' for p in ms_ssim_ema]

        print(f'[{os.path.basename(path):>{fiename_maxlen}}] ')
        print('eval: ', psnr, ms_ssim)
        print('ema : ', psnr_ema, ms_ssim_ema)
        for i, (xr_level, xr_level_ema) in enumerate(zip(x_recon, x_recon_ema)):
            save_image(xr_level.detach(), os.path.join(example_dir, f'{filename}_{model.itr:08}_level{i}.png'))
            save_image(xr_level_ema.detach(), os.path.join(example_dir, f'{filename}_{model.itr:08}_level{i}_ema.png'))

    else:
        x = x.cuda()

        t0 = time()

        z, z_shape = model.encode(x)
        x_recon = model.decode(z, shape=z_shape)
        z_ema, z_ema_shape = model.encode_ema(x)
        x_recon_ema = model.decode_ema(z_ema, shape=z_ema_shape)

        elapsed_t = time() - t0
        x_size = x.shape[-1] * x.shape[-2]
        bpp = len(z) * 8 / x_size
        bpp_ema = len(z_ema) * 8 / x_size

        z_bytes = compressor.decompress(z)
        z_np = np.frombuffer(z_bytes, dtype=np.int8).reshape(z_shape).copy()
        z_ema_bytes = compressor.decompress(z_ema)
        z_np_ema = np.frombuffer(z_ema_bytes, dtype=np.int8).reshape(z_ema_shape).copy()
        z_dict = dict(Counter(z_np.flatten()))
        z_ema_dict = dict(Counter(z_np_ema.flatten()))

        for k in z_dict:
            z_dict[k] = round(z_dict[k] / np.prod(z_shape), 2)
            z_ema_dict[k] = round(z_ema_dict[k] / np.prod(z_shape), 2)

        psnr = PSNR(x, x_recon)
        psnr_ema = PSNR(x, x_recon_ema)
        ms_ssim = MS_SSIM(x, x_recon)
        ms_ssim_ema = MS_SSIM(x, x_recon_ema)
        print(f'[{os.path.basename(path):>{fiename_maxlen}}] '
              f'{len(z):6}B {bpp:.4f}bpp {psnr[0]:.4f}, {ms_ssim[0]:.4f}' # {str(z_dict):>47}, '
              f'{len(z_ema):6}B {bpp_ema:.4f}bpp, {psnr_ema[0]:.4f}, {ms_ssim_ema[0]:.4f} ' # {str(z_ema_dict):>47} '
              f'({elapsed_t:>.4f}s)')

        save_image(x_recon.squeeze().detach(), os.path.join(example_dir, f'{filename}_{model.itr:08}_{bpp}.png'))
        save_image(x_recon_ema.squeeze().detach(), os.path.join(example_dir, f'{filename}_{model.itr:08}_{bpp_ema}_ema.png'))

