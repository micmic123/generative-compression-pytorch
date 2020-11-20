import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import copy2


threshold_size = 960
threshold_s = 0.9
threshold_v = 0.8
target = '/data/micmic123/openimage/f960'
p1 = '/data/micmic123/openimage/validation/*'
p2 = '/data/micmic123/openimage/challenge2018/*'
paths = glob(p1) + glob(p2)
os.makedirs(target, exist_ok=True)
for path in tqdm(paths):
    img = cv2.imread(path)
    H, W, C = img.shape
    h, s, v = cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    if max(H, W) < threshold_size or s > threshold_s or v > threshold_v:
        continue
    copy2(path, target)
