import os
import argparse
from time import time
from datetime import datetime
import torch
from tensorboardX import SummaryWriter
from utils import init, save_grid, lr_schedule, write_loss
from dataset import get_dataloader
from models.trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file path', type=str)
parser.add_argument('--name', help='result dir name', default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"), type=str)
parser.add_argument('--device', help='CUDA_VISIBLE_DEVICES number', default='3', type=str)
parser.add_argument('--resume', help='snapshot path', type=str)
parser.add_argument('--multigpus', type=str, default='')
args = parser.parse_args()


if not args.config:
    if args.resume:
        dir_path = '/'.join(args.resume.split('/')[:-2])
        args.config = os.path.join(dir_path, 'config.yaml')
    else:
        args.config = './configs/config.yaml'

base_dir, snapshot_dir, output_dir, log_path, config = init(args)
train_writer = SummaryWriter(base_dir)
train_dataloader, image_num, test_dataloader = get_dataloader(config)
test_loader = iter(test_dataloader)
config['image_num'] = image_num

if args.multigpus:
    args.device = args.multigpus # ','.join([str(n) for n in args.multigpus])
os.environ['CUDA_VISIBLE_DEVICES'] = args.device

print('[device]', args.device)
print('[config]', args.config)
msg = f'======================= {args.name} ======================='
print(msg)
for k, v in config.items():
    print(f'  {k}: ', v)
print('='*len(msg))
print()


trainer = Trainer(config)
trainer.cuda()
if args.resume:
    trainer.load(args.resume)

if args.multigpus:
    ngpus = torch.cuda.device_count()
    print(f'Number of GPUs: {ngpus}')
    trainer.model = torch.nn.DataParallel(trainer.model, device_ids=range(ngpus), output_device=1)
    trainer.multigpus = True


while True:
    update_D = 1
    for x in train_dataloader:
        t0 = time()

        if update_D == 1:
            loss_D_real, loss_D_fake, loss_D = trainer.D_update(x)
            update_D *= -1
            continue
        loss_recon, loss_fm, loss_G_adv, loss_vgg, loss_grad, loss_G = trainer.G_update(x)  # , mask_size
        update_D *= -1

        torch.cuda.synchronize()
        elapsed_t = time() - t0
        trainer.itr += 1

        if trainer.itr % config['lr_shedule_step'] == 0:
            lr_schedule(trainer.encoder_opt, config['lr_encoder'])
            lr_schedule(trainer.decoder_opt, config['lr_decoder'])
            lr_schedule(trainer.dis_opt, config['lr_dis'])
            if trainer.has_controller:
                lr_schedule(trainer.controller_opt, config['lr_dis'])

        if trainer.itr % config['log_itr'] == 0:
            write_loss(trainer.itr, trainer, train_writer)

        if trainer.itr % config['log_print_itr'] == 0:
            print(f'[{trainer.itr:>6}] recon={loss_recon:>.4f} | fm={loss_fm:>.4f} | G_adv={loss_G_adv:>.4f} | '
                  f'vgg={loss_vgg:>.4f} | grad={loss_grad:>.4f} | G={loss_G:>.4f} | D_real={loss_D_real:>.4f} | '
                  f'D_fake={loss_D_fake:>.4f} | D={loss_D:>.4f} {1:>3} ({elapsed_t:>.2f}s)')  # mask_size

        if trainer.itr % config['image_save_itr'] == 0:
            x_train = x[:config['batchsize_test']]
            x_train, x_train_recon, x_train_recon_ema = trainer.test(x_train)
            out = torch.cat([x_train.detach(), x_train_recon.detach(), x_train_recon_ema.detach()], dim=0)
            save_grid(out, f'{output_dir}/{trainer.itr:08}_train.png', nrow=4)

            try:
                x_test, size = next(test_loader)
            except StopIteration:
                test_loader = iter(test_dataloader)
                x_test, size = next(test_loader)
            x_test = x_test.cuda()
            x_test, x_test_recon, x_test_recon_ema = trainer.test(x_test)
            out = torch.cat([x_test.detach(), x_test_recon.detach(), x_test_recon_ema.detach()], dim=0)
            save_grid(out, f'{output_dir}/{trainer.itr:08}_test.png', nrow=4)

            z, z_shape = trainer.encode(x_train[0].unsqueeze(0))
            z_test, z_ema_shape = trainer.encode(x_test[0].unsqueeze(0))

            if not (config['mask'] or config['controller']):
                print(f'x_train[0]: {len(z)}bytes, x_test[0]: {len(z_test)}bytes')

        if trainer.itr % config['snapshot_save_itr'] == 0:
            trainer.save(snapshot_dir, f'itr_{trainer.itr:08}.pt')
        elif trainer.itr % config['snapshot_latest_save_itr'] == 0:
            trainer.save(snapshot_dir, 'latest.pt')
