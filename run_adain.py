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
    parser.add_argument('--style_config', type=str, default='config.json')
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
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--loss_momentum', type=float, default=0.1)
    parser.add_argument('--loss_ratio_thershold', type=float, default=0.1)
    parser.add_argument('--rec_loss', action='store_true')
    parser.add_argument('--style_loss_scale', type=float, default=1)
    parser.add_argument('--rec_step', type=int, default=1)

    return parser.parse_args()

def cal_step(args, model, c_imgs, s_imgs):
    _, _, norm_feats, t_imgs = model(c_imgs, s_imgs, return_hidden=True)
    s_feats = model.enc(s_imgs, return_last=False)
    t_feats = model.enc(t_imgs, return_last=False)
    rec_c_imgs = model.dec(model.enc(c_imgs))
    rec_s_imgs = model.dec(model.enc(s_imgs))
    rec_c_loss = F.mse_loss(rec_c_imgs, c_imgs)
    rec_s_loss = F.mse_loss(rec_s_imgs, s_imgs)
    c_loss = F.mse_loss(t_feats[-1], norm_feats)
    s_loss = 0
    for t_feat, s_feat in zip(t_feats, s_feats):
        t_feat_mean, t_feat_std = model.adain.cal_mean_std(t_feat)
        s_feat_mean, s_feat_std = model.adain.cal_mean_std(s_feat)
        s_loss += F.mse_loss(t_feat_mean, s_feat_mean)
        s_loss += F.mse_loss(t_feat_std, s_feat_std)
    return c_loss, s_loss, rec_c_loss, rec_s_loss

def main(args):
    trainset, valset = utils.dataloader.split_style_dataset(
                        valid_ratio=args.valid_ratio,
                        content_dir=args.content_dir,
                        style_config=args.style_config,
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
    with open(os.path.join(args.checkpath, 'style.json'), 'w') as f:
        f.write(json.dumps(trainset.style_config, indent=2))


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

    pbar = tqdm(range(args.steps), ncols=100)
    train_c_loss = []
    train_s_loss = []
    train_rec_c_loss = []
    train_rec_s_loss = []
    bst_loss = float('inf')
    loss_ratio = torch.Tensor([1. for _ in range(4)]).to(args.device)
    for step in pbar:
        if step % len(train_loader) == 0:
            train_iter = iter(train_loader)
        train_c_imgs, train_s_imgs, train_s_tags = next(train_iter)
        train_c_imgs = train_c_imgs.to(args.device)
        train_s_imgs = train_s_imgs.to(args.device)
        c_loss, s_loss, rec_c_loss, rec_s_loss = cal_step(
                            args, model,
                            train_c_imgs,
                            train_s_imgs
                         )
        if args.rec_loss:
            if step % args.rec_step == 0:
                loss = loss_ratio[0] * c_loss \
                       + loss_ratio[1] * s_loss * args.style_loss_scale \
                       + loss_ratio[2] * rec_c_loss \
                       + loss_ratio[3] * rec_s_loss
            else:
                loss = (loss_ratio[0] * c_loss + loss_ratio[1] * s_loss * args.style_loss_scale) / (loss_ratio[0] + loss_ratio[1])
                rec_c_loss = rec_c_loss.detach()
                rec_s_loss = rec_s_loss.detach()
        else:
            loss = c_loss + args.lamb * s_loss
            rec_c_loss = rec_c_loss.detach()
            rec_s_loss = rec_s_loss.detach()
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_c_loss.append(c_loss.item())
        train_s_loss.append(s_loss.item())
        train_rec_c_loss.append(rec_c_loss.item())
        train_rec_s_loss.append(rec_s_loss.item())

        # Update Loss Ratio
        with torch.no_grad():
            new_ratio = torch.stack(
                            [
                                1 / c_loss.detach(),
                                1 / s_loss.detach(),
                                1 / rec_c_loss.detach(),
                                1 / rec_s_loss.detach()
                            ]
                        )
            new_ratio = new_ratio / new_ratio.sum()
            loss_ratio = (1 - args.loss_momentum) * loss_ratio \
                         + args.loss_momentum * new_ratio
            loss_ratio = torch.where(
                            loss_ratio > args.loss_ratio_thershold,
                            loss_ratio,
                            args.loss_ratio_thershold
                         )
            loss_ratio = 4 * loss_ratio / loss_ratio.sum()

        pbar.set_description(
            f'Lc {c_loss.item():.2e} Ls {s_loss.item():.2e} Lrecc {rec_c_loss.item():.2e} Lrecs {rec_s_loss.item():.2e}'
        )

        if step % args.log_interval == 0 and step != 0:
            val_c_loss = []
            val_s_loss = []
            val_rec_c_loss = []
            val_rec_s_loss = []
            for val_c_imgs, val_s_imgs, val_s_tags in val_loader:
                with torch.no_grad():
                    val_c_imgs = val_c_imgs.to(args.device)
                    val_s_imgs = val_s_imgs.to(args.device)
                    c_loss, s_loss, rec_c_loss, rec_s_loss = cal_step(
                        args,
                        model,
                        val_c_imgs,
                        val_s_imgs
                    )
                    val_c_loss.append(c_loss.item())
                    val_s_loss.append(s_loss.item())
                    val_rec_c_loss.append(rec_c_loss.item())
                    val_rec_s_loss.append(rec_s_loss.item())

            # Write Log
            with open(args.log_file, 'r') as f:
                logs = json.load(f)
            log = {
                'steps': step,
                'train_c_loss': np.mean(train_c_loss),
                'train_s_loss': np.mean(train_s_loss),
                'train_rec_c_loss': np.mean(train_rec_c_loss),
                'train_rec_s_loss': np.mean(train_rec_s_loss),
                'val_c_loss': np.mean(val_c_loss),
                'val_s_loss': np.mean(val_s_loss),
                'val_rec_c_loss': np.mean(val_rec_c_loss),
                'val_rec_s_loss': np.mean(val_rec_s_loss),
                'loss_ratio': loss_ratio.tolist(),
            }
            logs.append(log)
            train_c_loss = []
            train_s_loss = []
            train_rec_c_loss = []
            train_rec_s_loss = []
            with open(args.log_file, 'w') as f:
                f.write(json.dumps(logs, indent=2))
            pbar.write(str(log))

            if args.rec_loss:
                cur_loss = loss_ratio[0] * np.mean(val_c_loss) \
                           + loss_ratio[1] * np.mean(val_s_loss) * args.style_loss_scale \
                           + loss_ratio[2] * np.mean(val_rec_c_loss) \
                           + loss_ratio[3] * np.mean(val_rec_s_loss)
            else:
                cur_loss = np.mean(val_c_loss) + args.lamb * np.mean(val_s_loss)
            if cur_loss < bst_loss:
                bst_loss = cur_loss.item()
                torch.save(model, os.path.join(args.checkpath, 'best.pt'))
                pbar.write(f'Save Best Model at {step} with L {bst_loss}')
        else:
            with open(args.log_file, 'r') as f:
                logs = json.load(f)
            log = {
                'steps': step,
                'train_c_loss': np.mean(train_c_loss),
                'train_s_loss': np.mean(train_s_loss),
                'train_rec_c_loss': np.mean(train_rec_c_loss),
                'train_rec_s_loss': np.mean(train_rec_s_loss),
                'loss_ratio': loss_ratio.tolist(),
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
