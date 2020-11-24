import os
import argparse
from time import time
from glob import glob
from PIL import Image
from utils import save_image, get_config
from dataset import inference_transform
from models.model import Model


parser = argparse.ArgumentParser()
# parser.add_argument('--config', help='config file path', type=str)
parser.add_argument('--device', help='CUDA_VISIBLE_DEVICES number', default='3', type=str)
parser.add_argument('--img', help='image path', type=str)
parser.add_argument('--snapshot', help='snapshot path', type=str, required=True)
args = parser.parse_args()


base_path = '/'.join(args.snapshot.split('/')[:-2])
args.config = os.path.join(base_path, 'config.yaml')
example_dir = os.path.join(base_path, 'examples')
os.makedirs(example_dir, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
config = get_config(args.config)
if 'mask' not in config:
    config['mask'] = 0

print('[device]', args.device)
print('[config]', args.config)
msg = f'======================= {args.snapshot} ======================='
print(msg)
for k, v in config.items():
    print(f'  {k}: ', v)
print('='*len(msg))
print()

model = Model(config)
model.cuda()
model.load(args.snapshot)

if args.img:
    paths = [args.img]
else:
    paths = glob('./samples/*')
for path in paths:
    img = Image.open(path).convert('RGB')
    x = inference_transform(img)
    x = x.unsqueeze(0)
    x = x.cuda()

    t0 = time()

    z, z_shape = model.encode(x)
    x_recon = model.decode(z, shape=z_shape)
    z_ema, z_ema_shape = model.encode_ema(x)
    x_recon_ema = model.decode_ema(z_ema, shape=z_ema_shape)

    elapsed_t = time() - t0
    save_image(x_recon.squeeze().detach(), os.path.join(example_dir, f'{os.path.basename(path)}_{model.itr:08}.png'))
    save_image(x_recon_ema.squeeze().detach(), os.path.join(example_dir, f'{os.path.basename(path)}_{model.itr:08}_ema.png'))
    print(f'elapsed time: {elapsed_t:>.4f}s')
    print(f'z: {len(z)} bytes, z_ema: {len(z_ema)} bytes')

