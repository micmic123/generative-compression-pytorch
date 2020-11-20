import os
import pandas as pd
import glob
from shutil import move
from tqdm import tqdm
from random import random


base = '/data/micmic123/openimage/f960/'
target = '/data/micmic123/openimage/f960_testset/'
all_paths = glob.glob(f'{base}*')
paths = pd.read_csv('../data/trainset.csv')['path'].tolist()
paths = set(paths)
for path in tqdm(all_paths):
    if path in paths or random() < 0.5:
        continue
    move(path, os.path.join(target, os.path.basename(path)))
