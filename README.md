# Generative Compression Pytorch
Pytorch implementation for [Generative Adversarial Networks for Extreme Learned Image Compression](https://arxiv.org/abs/1804.02958).


## Environment
We tested our code in ubuntu 16.04, cuda 10.1, python 3.8.8 and Pytorch 1.5.1.

## Dataset
- Cityscapes dataset
    - https://www.cityscapes-dataset.com/
    
- OpenImage dataset
    - https://storage.googleapis.com/openimages/web/index.html

1. Make a directory structure for the datasets as follows.
```
├── your_dataset_root
    └── leftImg8bit
        |── train
        ├── val
        └── test
```
2. Run `scripts/get_paths.py`  with arguments of dataset paths.
    - `$ ./scripts/get_paths.py --trainset=your_dataset_root/leftImg8bit/train/*/* --testset=your_dataset_root/leftImg8bit/test/*/*`


## Training
### Configuration
You can change the configuration including hyperparameters by modifying `./configs/config.yaml` to train the model.


### Quick start
`$ python train.py --config=./configs/config.yaml --name=your_instance_name` \
The checkpoints of the model will be saved in `./results/your_instance_name/snapshots`.

### Resume from the checkpoint
`$ python train.py --resume=./results/your_instance_name/snapshots/your_snapshot_name.pt` \
By default, the original configuration of the checkpoint `./results/your_instance_name/config.yaml` will be used.

## Inference
`$ python inference.py --snapshot=./results/your_instance_name/snapshots/your_snapshot_name.pt --img=input_img_path`

