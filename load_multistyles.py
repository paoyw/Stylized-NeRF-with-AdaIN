import os
import json
import numpy as np

import load_blender
import load_llff

def load_blender_data(basedir, half_res=False, testskip=1):
    with open(os.path.join(basedir, 'config.json'), 'r') as f:
        metas = json.load(f)

    images_stk = []
    poses_stk = []
    render_poses_stk = []
    styles_stk = []
    train_split_stk = []
    val_split_stk = []
    test_split_stk = []
    accum_bs = 0

    for meta in metas:
        images, poses, render_poses, [H, W, focal], [train_split, val_split, test_split] = load_blender.load_blender_data(
            meta['out_dir'], half_res=half_res, testskip=testskip,
        )
        images_stk.append(images)
        poses_stk.append(poses)
        render_poses.append(render_poses)
        train_split_stk.append(train_split + accum_bs)
        val_split_stk.append(val_split + accum_bs)
        test_split_stk.append(test_split + accum_bs)

        bs = imgs.shape[0]
        accum_bs += bs
        styles = np.load(os.path.join(meta['out_dir'], 'style.npy'))
        styles = np.stack([styles for _ in range(bs)])

    images_stk = np.concatenate(images_stk, axis=0)
    poses_stk = np.concatenate(poses_stk, axis=0)
    render_poses_stk = np.concatenate(render_poses_stk, axis=0)
    styles_stk = np.concatenate(styles_stk, axis=0)
    return imgs_stk, poses_stk, render_poses_stk, styles_stk, [H, W, focal], [train_split_stk, val_split_stk, test_split_stk]

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    with open(os.path.join(basedir, 'config.json'), 'r') as f:
        metas = json.load(f)

    images_stk = []
    poses_stk = []
    bds_stk = []
    render_poses_stk = []
    i_test_stk = []
    accum_bs = 0

    styles_stk = []
    for meta in metas: 
        images, poses, bds, render_poses, i_test = load_llff.load_llff_data(
            meta['out_dir'],
            factor=factor,
            recenter=recenter,
            bd_factor=bd_factor,
            spherify=spherify,
        )
        images_stk.append(images)
        poses_stk.append(poses)
        bds_stk.append(bds)
        render_poses_stk.append(render_poses)
        i_test_stk.append(i_test + accum_bs)
        bs = images.shape[0]
        accum_bs += bs
        if factor != 0:
            style = np.load(
                        os.path.join(
                            meta['out_dir'],
                            f'images_{factor}',
                            'style.npy'
                        )
                    )
        else:
            style = np.load(
                        os.path.join(
                            meta['out_dir'],
                            'images',
                            'style.npy'
                        )
                    )
        styles = np.stack([style for _ in range(bs)])
        styles_stk.append(styles)

    images_stk = np.concatenate(images_stk, axis=0)
    poses_stk = np.concatenate(poses_stk, axis=0)
    bds_stk = np.concatenate(bds_stk, axis=0)
    render_poses_stk = np.concatenate(render_poses_stk, axis=0)
    styles_stk = np.concatenate(styles_stk, axis=0)
    i_test_stk = np.array(i_test_stk,)

    return images_stk, poses_stk, bds_stk, styles_stk, i_test_stk
