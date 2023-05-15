import os
from argparse import ArgumentParser
import json

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

import utils

def parser_args():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--llff_dir', type=str)
    parser.add_argument('--style_img', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

def main(args):
    os.makedirs(args.out_dir, exist_ok=args.overwrite)

    # Load style transfer model
    model = torch.load(args.model, map_location=args.device)

    # Transformation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    o_tfm = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(mean=mean, std=std)
            ])
    s_tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    # Load sytle image
    s_img = Image.open(args.style_img).convert('RGB')
    s_img = s_tfm(s_img).unsqueeze(dim=0).to(args.device)
    s_sz = [s_img.size(-2), s_img.size(-1)]
    save_image(
        utils.dataloader.denorm(
            s_img.squeeze().to('cpu'),
        ),
        os.path.join(args.out_dir, 'style.jpg'),
    )
    
    os.system(f'cp -r {args.llff_dir}/* {args.out_dir}')

    def style_transfer_dir(direct):
        o_dir = os.path.join(args.llff_dir, direct)
        t_dir = os.path.join(args.out_dir, direct)
        fnames = os.listdir(o_dir)
        fnames = [
            fname for fname in fnames \
            if fname.endswith('jpg') \
                or  fname.endswith('JPG') \
                or  fname.endswith('png') \
                or fname.endswith('PNG')
        ]
        for fname in tqdm(fnames, ncols=50):
            o_fname = os.path.join(o_dir, fname)
            t_fname = os.path.join(t_dir, fname)
            c_img = Image.open(o_fname).convert('RGB')
            c_img = o_tfm(c_img).unsqueeze(dim=0).to(args.device)
            c_sz = [c_img.size(-2), c_img.size(-1)]
            if c_sz[0] / c_sz[1] > s_sz[0] / s_sz[1]:
                scale_factor = c_sz[0] / s_sz[0]
            else:
                scale_factor = c_sz[1] / s_sz[1]
            scale_size = [
                int(np.ceil(scale_factor * s_sz[0])),
                int(np.ceil(scale_factor * s_sz[1]))
            ]
            s_img_tfm = F.interpolate(
                            s_img,
                            size=scale_size,
                            mode='bilinear',
                        )
            s_img_tfm =  transforms.functional.center_crop(
                            s_img_tfm,
                            output_size=c_sz,
                         )
            t_img = model(c_img, s_img_tfm, return_hidden=False)
            t_img = utils.dataloader.denorm(t_img.squeeze().to('cpu'))
            save_image(t_img, t_fname)

    # img_directs = ['images', 'images_4', 'images_8']
    img_directs = ['images_4', 'images_8']
    with torch.no_grad():
        for direct in img_directs:
            style_transfer_dir(direct)

if __name__ == '__main__':
    args = parser_args()
    main(args)
