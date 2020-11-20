import os
import argparse
from time import time
from datetime import datetime
import torch
from utils import get_config, make_dir, save_image
from dataset import get_dataloader
from models.model import Model


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file path', type=str)
parser.add_argument('--name', help='result dir name', default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"), type=str)
parser.add_argument('--device', help='CUDA_VISIBLE_DEVICES number', default='3', type=str)
parser.add_argument('--resume', help='snapshot path', type=str)
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.device
base_dir, snapshot_dir, output_dir, log_path = make_dir(args)

if not args.config:
    if args.resume:
        args.config = os.path.join(args.resume, 'config.yaml')
    else:
        args.config = './configs/config.yaml'

config = get_config(args.config)
train_dataloader, image_num = get_dataloader(config)
config['image_num'] = image_num

print(f'VISIBLE CUDA DEVICE: {args.device}')
print('============== config ==============')
print('[path]', args.config)
for k, v in config.items():
    print(f'{k}: ', v)
print('====================================')

itr = 0
model = Model(config)
if args.resume:
    itr = model.load(args.resume)
model.cuda()


while True:
    for x, idx in train_dataloader:
        x, idx = x.cuda(), idx.cuda()
        t0 = time()

        loss_D_real, loss_D_fake, loss_D = model.D_update(x, idx)
        loss_recon, loss_fm, loss_G_adv, loss_G = model.G_update(x, idx)

        elapsed_t = time() - t0
        itr += 1

        if (itr) % config['log_itr'] == 0:
            print(f'[{itr:>6}] recon={loss_recon:>.4f} | fm={loss_fm:>.4f} | G_adv={loss_G_adv:>.4f} | G={loss_G:>.4f} | '
                  f'D_real={loss_D_real:>.4f} | D_fake={loss_D_fake:>.4f} | D={loss_D:>.4f} ({elapsed_t:>.2f}s)')

        if (itr) % config['image_save_itr'] == 0:
            test_x = x[:4]
            test_idx = idx[:4]
            test_x_recon, test_x_recon_ema = model.test(test_idx)
            out = torch.cat([test_x.detach(), test_x_recon.detach(), test_x_recon_ema.detach()], dim=0)
            save_image(out, f'{output_dir}/{itr}.png', nrow=4)


