import pandas as pd
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='path for dataset', default=f'/data/micmic123/openimage/f960/')
args = parser.parse_args()

data_dir = './data'
os.makedirs('./data', exist_ok=True)
dataset_path = glob.glob(os.path.join(args.dataset, '*.png'))
dataset_path += glob.glob(os.path.join(args.dataset, '*.jpg'))
pd.DataFrame(dataset_path, columns=['path']).to_csv(f'{data_dir}/trainset.csv', index=False)
