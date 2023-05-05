import os
import json
import random

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def denorm(imgs):
    mean_t = torch.Tensor(
                mean, device=imgs.device
             ).view(3, *[1 for _ in range(len(imgs) - 1)])
    std_t = torch.Tensor(
                std, device=imgs.device
            ).view(3, *[1 for _ in range(len(imgs) - 1)])
    return std_t * imgs + mean_t

def split_style_dataset(
    valid_ratio=0.1,
    content_dir='data/train2014/',
    style_dir='data/style/',
):
    content_files = [
        os.path.join(content_dir, file) for file in os.listdir(content_dir)
    ]
    train_files, val_files = train_test_split(
                                content_files,
                                test_size=valid_ratio
                              )
    trainset = StyleDataset(content_files=train_files, style_dir=style_dir)
    valset = StyleDataset(content_files=val_files, style_dir=style_dir)
    return trainset, valset

class StyleDataset(Dataset):
    def __init__(
        self,
        content_files,
        style_dir='data/style/',
    ):
        self.content_files = content_files
        self.style_dir = style_dir
        with open(os.path.join(style_dir, 'config.json'), 'r') as f:
            self.style_config = json.load(f)
        for key in self.style_config.keys():
            self.style_config[key] = os.path.join(
                                         style_dir,
                                         self.style_config[key]
                                     )

        self.tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=512, antialias=True),
                transforms.RandomCrop(size=(256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )

    def __getitem__(self, index):
        '''
        content image, style image, style tag
        '''
        content_file = self.content_files[index]
        content_img = Image.open(content_file).convert('RGB')
        content_img = self.tfm(content_img)

        style_tag = random.choice(tuple(self.style_config.keys()))
        style_img = Image.open(
                        self.style_config[style_tag]
                    ).convert('RGB')
        style_img = self.tfm(style_img)
        return content_img, style_img, style_tag

        
    def __len__(self):
        return len(self.content_files)
