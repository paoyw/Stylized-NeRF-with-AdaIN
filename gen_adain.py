import os
from argparse import ArgumentParser

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

import utils

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--content_image', type=str, required=True)
    parser.add_argument('--style_image', type=str, required=True)
    parser.add_argument('--transfer_image', type=str, required=True)
    return parser.parse_args()

def main(args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 1024), antialias=False),
            transforms.Normalize(mean=mean, std=std)
          ])
    content_img = Image.open(args.content_image).convert('RGB')
    content_img = tfm(content_img).unsqueeze(dim=0).to(args.device)
    style_img = Image.open(args.style_image).convert('RGB')
    style_img = tfm(style_img).unsqueeze(dim=0).to(args.device)

    model = torch.load(args.model, map_location=args.device)
    with torch.no_grad():
        c_feats, s_feats, norm_feats, trans_img = model(
                                                    content_img,
                                                    style_img,
                                                    return_hidden=True
                                                  )
        t_feats = model.enc(trans_img) 
        rec_c_img = model.dec(c_feats)
        rec_s_img = model.dec(s_feats)

        trans_img = utils.dataloader.denorm(
                        trans_img.squeeze().to('cpu')
                    )
        rec_c_img = utils.dataloader.denorm(
                        rec_c_img.squeeze().to('cpu')
                    )
        rec_s_img = utils.dataloader.denorm(
                        rec_s_img.squeeze().to('cpu')
                    )
        cat_img = torch.cat(
            [
                utils.dataloader.denorm(
                    content_img.squeeze().to('cpu'),
                ),
                utils.dataloader.denorm(
                    style_img.squeeze().to('cpu'),
                ),
                trans_img,
                rec_c_img,
                rec_s_img,
            ],
            dim=2
         )
    save_image(cat_img, args.transfer_image)

if __name__ == '__main__':
    args = parse_args()
    main(args)

