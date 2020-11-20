import pandas as pd
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trainset', help='path for dataset', default=f'/data/micmic123/openimage/f960/')
parser.add_argument('--testset', help='path for dataset', default=f'/data/micmic123/openimage/f960_testset/')
args = parser.parse_args()

data_dir = '../data'
os.makedirs(data_dir, exist_ok=True)
dataset_path = glob.glob(os.path.join(args.trainset, '*.png'))
dataset_path += glob.glob(os.path.join(args.trainset, '*.jpg'))
pd.DataFrame(dataset_path, columns=['path']).to_csv(f'{data_dir}/trainset.csv', index=False)

dataset_path = glob.glob(os.path.join(args.testset, '*.png'))
dataset_path += glob.glob(os.path.join(args.testset, '*.jpg'))
pd.DataFrame(dataset_path, columns=['path']).to_csv(f'{data_dir}/testset.csv', index=False)
