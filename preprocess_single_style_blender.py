import os
from argparse import ArgumentParser
import json

from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

import utils

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--blender_dir', type=str)
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
                transforms.Resize(800, antialias=False),
                transforms.CenterCrop((800, 800)),
                transforms.Normalize(mean=mean, std=std)
            ])

    # Load style image
    s_img = Image.open(args.style_img).convert('RGB')
    s_img = s_tfm(s_img).unsqueeze(dim=0).to(args.device)
    save_image(
        utils.dataloader.denorm(
            s_img.squeeze().to('cpu'),
        ),
        os.path.join(args.out_dir, 'style.jpg'),
    )

    def style_transfer_dir(pos):
        os.makedirs(
            os.path.join(args.out_dir, pos),
            exist_ok=args.overwrite
        )
        with open(
                os.path.join(
                    args.blender_dir,
                    f'transforms_{pos}.json'
                ),
                'r'
             )  as f:
            train_meta = json.load(f)
        with open(
                os.path.join(
                    args.out_dir,
                    f'transforms_{pos}.json',
                ),
                'w'
             ) as f:
            f.write(json.dumps(train_meta, indent=2))
        for frame in tqdm(train_meta['frames'], ncols=50):
            o_fname = os.path.join(
                        args.blender_dir,
                        frame['file_path']+'.png'
                      )
            s_fname = os.path.join(
                        args.out_dir,
                        frame['file_path']+'.png'
                      )
            c_img = Image.open(o_fname).convert('RGB')
            c_img = o_tfm(c_img).unsqueeze(dim=0).to(args.device)
            t_img = model(c_img, s_img, return_hidden=False)
            t_img = utils.dataloader.denorm(t_img.squeeze().to('cpu'))
            save_image(t_img, s_fname)
    style_transfer_dir('train')
    style_transfer_dir('val')
    style_transfer_dir('test')


if __name__ == '__main__':
    args = parse_args()
    main(args)
