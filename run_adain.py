import os
from argparse import ArgumentParser
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils
import models

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpath', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', type=str, default='log.json')
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=10000)

    # Data
    parser.add_argument('--content_dir', type=str, default='data/train2014/')
    parser.add_argument('--style_dir', type=str, default='data/style/')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    # Hyper-parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--valid_ratio', type=float, default=0.05)

    parser.add_argument('--optimizer', type=str, choices=['adam', 'adam_w', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)

    return parser.parse_args()

def cal_step(args, model, c_imgs, s_imgs):
    _, _, norm_feats, t_imgs = model(c_imgs, s_imgs)
    s_feats = model.enc(s_imgs, return_last=False)
    t_feats = model.enc(t_imgs, return_last=False)
    c_loss = F.mse_loss(t_feats[-1], norm_feats)
    s_loss = 0
    for t_feat, s_feat in zip(t_feats, s_feats):
        t_feat_mean, t_feat_std = model.adain.cal_mean_std(t_feat)
        s_feat_mean, s_feat_std = model.adain.cal_mean_std(s_feat)
        s_loss += F.mse_loss(t_feat_mean, s_feat_mean)
        s_loss += F.mse_loss(t_feat_std, s_feat_std)
    return c_loss, s_loss

def main(args):
    trainset, valset = utils.dataloader.split_style_dataset(
                        valid_ratio=args.valid_ratio,
                        content_dir=args.content_dir,
                        style_dir=args.style_dir,
                       )
    train_loader = torch.utils.data.DataLoader(
                     trainset,
                     batch_size=args.batch_size,
                     shuffle=True,
                     num_workers=args.num_workers,
                     drop_last=True,
                   )
    val_loader = torch.utils.data.DataLoader(
                     valset,
                     batch_size=args.batch_size,
                     shuffle=False,
                     num_workers=args.num_workers,
                     drop_last=True,
                 )

    model = models.models.StyleTransferAdaIN()
    model = model.to(args.device)

    if args.optimizer == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam_w':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    pbar = tqdm(range(args.steps), ncols=80)
    train_c_loss = []
    train_s_loss = []
    bst_loss = float('inf')
    for step in pbar:
        if step % len(train_loader) == 0:
            train_iter = iter(train_loader)
        train_c_imgs, train_s_imgs, train_s_tags = next(train_iter)
        train_c_imgs = train_c_imgs.to(args.device)
        train_s_imgs = train_s_imgs.to(args.device)
        c_loss, s_loss = cal_step(args, model, train_c_imgs, train_s_imgs)
        loss = args.alpha * c_loss + args.beta * s_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_c_loss.append(c_loss.item())
        train_s_loss.append(s_loss.item())
        pbar.set_description(f'Lc {c_loss.item():.2e} Ls {s_loss.item():.2e}')

        if step % args.log_interval == 0 and step != 0:
            val_c_loss = []
            val_s_loss = []
            for val_c_imgs, val_s_imgs, val_s_tags in val_loader:
                with torch.no_grad():
                    val_c_imgs = val_c_imgs.to(args.device)
                    val_s_imgs = val_s_imgs.to(args.device)
                    c_loss, s_loss = cal_step(args, model, val_c_imgs, val_s_imgs)
                    val_c_loss.append(c_loss.item())
                    val_s_loss.append(s_loss.item())

            # Write Log
            with open(args.log_file, 'r') as f:
                logs = json.load(f)
            log = {
                'steps': step,
                'train_c_loss': np.mean(train_c_loss),
                'train_s_loss': np.mean(train_s_loss),
                'val_c_loss': np.mean(val_c_loss),
                'val_s_loss': np.mean(val_s_loss),
            }
            logs.append(log)
            train_c_loss = []
            trian_s_loss = []
            with open(args.log_file, 'w') as f:
                f.write(json.dumps(logs, indent=2))
            pbar.write(str(log))

            if args.alpha * np.mean(val_c_loss) + args.beta * np.mean(val_s_loss) < bst_loss:
                bst_loss = args.alpha * np.mean(val_c_loss) + args.beta * np.mean(val_s_loss)
                torch.save(model, os.path.join(args.checkpath, 'best.pt'))
                pbar.write(f'Save Best Model at {step} with L {bst_loss}')
        else:
            with open(args.log_file, 'r') as f:
                logs = json.load(f)
            log = {
                'steps': step,
                'train_c_loss': np.mean(train_c_loss),
                'train_s_loss': np.mean(train_s_loss),
            }
            logs.append(log)
            with open(args.log_file, 'w') as f:
                f.write(json.dumps(logs, indent=2))

        if step % args.save_interval == 0 and step != 0:
            torch.save(model, os.path.join(args.checkpath, f'step_{step}.pt'))

if __name__ == '__main__':
    args = parse_args()
    utils.commons.same_seed(args.seed)
    utils.commons.create_checkpath(args)
    main(args)
