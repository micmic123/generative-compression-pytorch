import os
import math
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


SCALE_MIN = 0.5
SCALE_MAX = 1
CROP_SIZE = 256


class OpenImageDataset(Dataset):
    def __init__(self, path, transform=None, mode='train'):
        df = pd.read_csv(path)
        self.paths = sorted(df['path'].tolist())
        self.transform = transform
        self.mode = mode
        print(f'[{mode}set] {len(self.paths)} images.')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # if H < 960 or W > 960: discard
        img = Image.open(self.paths[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.mode == 'train':
            return img
        elif self.mode == 'test':
            return img, os.path.getsize(self.paths[idx])


# Resize the input PIL Image to the given size.
class ResizeLongerTo:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        size = self.size
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
        if not (isinstance(size, int)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        w, h = img.size
        if (w <= h and h == size) or (h <= w and w == size):
            return img
        if w >= h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), Image.BILINEAR)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), Image.BILINEAR)


class RandomResize:
    def __init__(self, scale_min=0.75, scale_max=1, size_min=256, interpolation=Image.BILINEAR):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.size_min = size_min
        self.interpolation = interpolation

    def __call__(self, img):
        # img: PIL image
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        W, H = img.size
        shortest_side_length = min(H, W)
        minimum_scale_factor = float(self.size_min) / float(shortest_side_length)
        scale_low = max(minimum_scale_factor, self.scale_min)
        scale_high = max(scale_low, self.scale_max)
        scale = np.random.uniform(scale_low, scale_high)
        img = img.resize((math.ceil(scale * H), math.ceil(scale * W)), self.interpolation)

        return img


base_transform = transforms.Compose([
    # ResizeLongerTo(768),
    RandomResize(SCALE_MIN, SCALE_MAX, CROP_SIZE),
    transforms.RandomCrop(CROP_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
])


inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
])

def get_dataloader(config):
    data_transforms = {
        'train': base_transform
    }

    train_dataset = OpenImageDataset(config['trainset'], data_transforms['train'], mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True, num_workers=config['worker_num'])

    test_dataset = OpenImageDataset(config['testset'], data_transforms['train'], mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=config['batchsize_test'], shuffle=True)

    return train_dataloader, len(train_dataset), test_dataloader
