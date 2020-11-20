import os
import argparse
from time import time
from datetime import datetime
import torch
from tensorboardX import SummaryWriter
from utils import init, save_image, lr_schedule, write_loss
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

print('[device]', args.device)
print('[config]', args.config)
msg = f'======================= {args.name} ======================='
print(msg)
for k, v in config.items():
    print(f'  {k}: ', v)
print('='*len(msg))
print()

itr = 0
model = Model(config)
model.cuda()
if args.resume:
    itr = model.load(args.resume)

while True:
    update_D = 1
    for x in train_dataloader:
        x = x.cuda()
        t0 = time()

        if update_D == 1:
            loss_D_real, loss_D_fake, loss_D = model.D_update(x)
            update_D *= -1
            continue
        loss_recon, loss_fm, loss_G_adv, loss_vgg, loss_G = model.G_update(x)
        update_D *= -1

        elapsed_t = time() - t0
        itr += 1

        if itr % config['lr_shedule_step'] == 0:
            lr_schedule(model.encoder_opt, config['lr_encoder'])
            lr_schedule(model.decoder_opt, config['lr_decoder'])
            lr_schedule(model.dis_opt, config['lr_dis'])

        if itr % config['log_itr'] == 0:
            write_loss(itr, model, train_writer)

        if itr % config['log_print_itr'] == 0:
            print(f'[{itr:>6}] recon={loss_recon:>.4f} | fm={loss_fm:>.4f} | G_adv={loss_G_adv:>.4f} | '
                  f'vgg={loss_vgg:>.4f} | G={loss_G:>.4f} | D_real={loss_D_real:>.4f} | D_fake={loss_D_fake:>.4f} | '
                  f'D={loss_D:>.4f} ({elapsed_t:>.2f}s)')

        if itr % config['image_save_itr'] == 0:
            x_train = x[:4]
            x_train_recon, x_train_recon_ema = model.test(x_train)
            out = torch.cat([x_train.detach(), x_train_recon.detach(), x_train_recon_ema.detach()], dim=0)
            save_image(out, f'{output_dir}/{itr:>6}_train.png', nrow=4)

            x_test, size = next(test_loader).cuda()
            x_test_recon, x_test_recon_ema = model.test(x_test)
            out = torch.cat([x_test.detach(), x_test_recon.detach(), x_test_recon_ema.detach()], dim=0)
            save_image(out, f'{output_dir}/{itr:>6}_test.png', nrow=4)

            print(f'x_train[0] bytes: {len(model.encode(x_train[0].unsqueeze(0)))}',
                  f'x_test[0] bytes: {len(model.encode(x_train[0].unsqueeze(0)))}')

        if itr % config['snapshot_save_itr'] == 0:
            model.save(snapshot_dir, itr)
