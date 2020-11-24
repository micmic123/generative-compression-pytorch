import os
import argparse
import atexit
from time import time
from datetime import datetime
import torch
from tensorboardX import SummaryWriter
from utils import init, save_grid, lr_schedule, write_loss
from dataset import get_dataloader
from models.model import Model


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file path', type=str)
parser.add_argument('--name', help='result dir name', default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"), type=str)
parser.add_argument('--device', help='CUDA_VISIBLE_DEVICES number', default='3', type=str)
parser.add_argument('--resume', help='snapshot path', type=str)
args = parser.parse_args()


if not args.config:
    if args.resume:
        dir_path = '/'.join(args.resume.split('/')[:-2])
        args.config = os.path.join(dir_path, 'config.yaml')
    else:
        args.config = './configs/config.yaml'
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
base_dir, snapshot_dir, output_dir, log_path, config = init(args)
train_writer = SummaryWriter(base_dir)
train_dataloader, image_num, test_dataloader = get_dataloader(config)
test_loader = iter(test_dataloader)
config['image_num'] = image_num
if config['mask'] == 1:
    config['C'] = config['C_level'][-1]

print('[device]', args.device)
print('[config]', args.config)
msg = f'======================= {args.name} ======================='
print(msg)
for k, v in config.items():
    print(f'  {k}: ', v)
print('='*len(msg))
print()


model = Model(config)
model.cuda()

if args.resume:
    model.load(args.resume)

# atexit.register(model.save, snapshot_dir)

while True:
    update_D = 1

    for x in train_dataloader:
        x = x.cuda()
        t0 = time()

        if update_D == 1:
            loss_D_real, loss_D_fake1, loss_D_fake2, loss_D = model.D_update(x)
            update_D *= -1
            continue
        loss_recon1, loss_recon2, \
        loss_fm1, loss_fm2, \
        loss_G_adv1, loss_G_adv2, \
        loss_vgg1, loss_vgg2, \
        loss_G = model.G_update(x)
        update_D *= -1

        elapsed_t = time() - t0
        model.itr += 1

        if model.itr % config['lr_shedule_step'] == 0:
            lr_schedule(model.encoder_opt, config['lr_encoder'])
            lr_schedule(model.decoder_opt, config['lr_decoder'])
            lr_schedule(model.dis_opt, config['lr_dis'])

        if model.itr % config['log_itr'] == 0:
            write_loss(model.itr, model, train_writer)

        if model.itr % config['log_print_itr'] == 0:
            print(f'[{model.itr:>6}] '
                  f'recon={loss_recon1:>.4f} | recon={loss_recon2:>.4f} | '
                  f'fm={loss_fm1:>.4f} | fm={loss_fm2:>.4f} | '
                  f'G_adv={loss_G_adv1:>.4f} | G_adv={loss_G_adv2:>.4f} | '
                  f'vgg={loss_vgg1:>.4f} | vgg={loss_vgg1:>.4f} | '
                  f'D_real={loss_D_real:>.4f} | '
                  f'D_fake={loss_D_fake1:>.4f} | D_fake={loss_D_fake2:>.4f} | '
                  f'D={loss_D:>.4f} ({elapsed_t:>.2f}s)')

        if model.itr % config['image_save_itr'] == 0:
            x_train = x[:config['batchsize_test']]
            x, x_recon1, x_recon2, x_recon1_ema, x_recon2_ema = model.test(x_train)
            out = torch.cat([x_train.detach(), x_recon1.detach(), x_recon2.detach(), x_recon1_ema.detach(), x_recon2_ema.detach()], dim=0)
            save_grid(out, f'{output_dir}/{model.itr:08}_train.png', nrow=4)

            try:
                x_test, size = next(test_loader)
            except StopIteration:
                test_loader = iter(test_dataloader)
                x_test, size = next(test_loader)
            x_test = x_test.cuda()
            x, x_recon1, x_recon2, x_recon1_ema, x_recon2_ema = model.test(x_test)
            out = torch.cat([x_test.detach(), x_recon1.detach(), x_recon2.detach(), x_recon1_ema.detach(), x_recon2_ema.detach()], dim=0)
            save_grid(out, f'{output_dir}/{model.itr:08}_test.png', nrow=4)

            # z, z_shape = model.encode(x_train[0].unsqueeze(0))
            # z_test, z_ema_shape = model.encode(x_test[0].unsqueeze(0))
            #
            # if not config['mask']:
            #     print(f'x_train[0]: {len(z)}bytes, x_test[0]: {len(z_test)}bytes')

        if model.itr % config['snapshot_save_itr'] == 0:
            model.save(snapshot_dir, f'itr_{model.itr:08}.pt')
        elif model.itr % config['snapshot_latest_save_itr'] == 0:
            model.save(snapshot_dir, 'latest.pt')
