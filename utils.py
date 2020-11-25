import os
import yaml
import time
from shutil import copy2
from torchvision import utils as vutils
import numpy as np
from PIL import Image


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def init(args):
    base_dir = f'./results/{args.name}'
    snapshot_dir = os.path.join(base_dir, 'snapshots')
    output_dir = os.path.join(base_dir, 'outputs')
    log_path = os.path.join(base_dir, 'logs.log')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    config = get_config(args.config)
    if config['mask'] == 1 or config['controller'] == 1:
        config['C'] = config['C_level'][-1]
    copy2(args.config, os.path.join(base_dir, 'config.yaml'))

    return base_dir, snapshot_dir, output_dir, log_path, config


def write_loss(itr, model, train_writer):
    members = [attr for attr in dir(model)
               if ((not callable(getattr(model, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    for m in members:
        train_writer.add_scalar(m, getattr(model, m), itr)


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