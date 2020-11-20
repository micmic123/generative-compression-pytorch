import os
import argparse
from time import time
from datetime import datetime
import torch
from utils import get_config, make_dir, save_image
from dataset import get_dataloader
from models.model import Model


parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', help='path for data path csv', default='./data/trainset.csv')
# parser.add_argument('--batchsize', help='batch size for training', default=0, type=int)
parser.add_argument('--config', help='config file', default='./configs/my_config.yaml', type=str)
parser.add_argument('--name', help='result dir name', default=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"), type=str)
parser.add_argument('--device', help='CUDA_VISIBLE_DEVICES NUMBER', default='3', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device

base_dir, snapshot_dir, output_dir, log_path = make_dir(args)
config = get_config(args.config)

train_dataloader, image_num = get_dataloader(config)
config['image_num'] = image_num
# if args.batchsize != 0:
#     config['batchsize'] = args.batchsize

model = Model(config)
model.cuda()

print(config)
print('table shape:', model.table.table.shape)

itr = 0
while True:
    for x, idx in train_dataloader:
        x, idx = x.cuda(), idx.cuda()
        t0 = time()
        if 0 in idx:
            with torch.no_grad():
                iii = torch.tensor([0]).cuda()
                print(idx)
                print(model.table.table[iii])
                print(model.table(iii))

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


