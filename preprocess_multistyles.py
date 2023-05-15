import os
from argparse import ArgumentParser
import json

import preprocess_single_style_blender
import preprocess_single_style_llff

class Subargs():
    def __init__(self):
        pass

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--content_dir', type=str)
    parser.add_argument('--style_imgs', type=str, nargs='+')
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

def main(args):
    os.makedirs(args.out_dir, exist_ok=args.overwrite)

    subargs = Subargs()
    subargs.overwrite = args.overwrite
    subargs.model = args.model
    subargs.device = args.device
    style_configs = []

    if args.dataset_type == 'blender':
        subargs.blender_dir =  args.content_dir
        for style_img in args.style_imgs:
            style_name = style_img.split('/')[-1].split('.')[0]
            subargs.out_dir = os.path.join(
                args.out_dir, style_name
            )
            subargs.style_img = style_img
            preprocess_single_style_blender.main(subargs)
            style_configs.append(
                {
                    'out_dir': subargs.out_dir,
                    'style_name': style_name,
                    'style_img': style_img,
                }
            )
    elif args.dataset_type == 'llff':
        subargs.llff_dir =  args.content_dir
        for style_img in args.style_imgs:
            style_name = style_img.split('/')[-1].split('.')[0]
            subargs.out_dir = os.path.join(
                args.out_dir, style_name
            )
            subargs.style_img = style_img
            preprocess_single_style_llff.main(subargs)
            style_configs.append(
                {
                    'out_dir': subargs.out_dir,
                    'style_name': style_name,
                    'style_img': style_img,
                }
            )
    else:
        raise NotImplementedError

    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(style_configs, indent=2))

if __name__ == '__main__':
    args = parse_args()
    main(args)
