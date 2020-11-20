import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OptimDataset(Dataset):
    def __init__(self, path, transform=None):
        df = pd.read_csv(path)
        self.paths = sorted(df['path'].tolist())
        self.transform = transform
        print(f'[trainset] {len(self.paths)} images.')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # if H < 960 or W > 960: discard
        image = Image.open(self.paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, idx


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


def get_dataloader(config):
    data_transforms = {
        'train': transforms.Compose([
            ResizeLongerTo(768),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
        ])
    }

    train_dataset = OptimDataset(config['dataset'], data_transforms['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True, num_workers=4)

    return train_dataloader, len(train_dataset)
