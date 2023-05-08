import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# NeRF
class PositionalEncoding(nn.Module):
    def __init__(self, L):
        '''
        Positional Encoding mapping R into R^{2L}
        '''
        super().__init__()
        self.L = L

    def forward(self, x):
        '''
        input:
            x: (N, F_in)
        output:
            gamma(x) = (
                cos(2^0*pi*x), ..., cos(2^{L-1}*pi*x)
                sin(2^0*pi*x), ...,sin(2^{L-1}*pi*x),
            )
            F_out = F_in * 2L
            outpupt: (N, F_out)
        '''
        cos_stk = torch.concat(
                    [
                        torch.cos((2**l)*torch.pi*x) \
                        for l in range(self.L)
                    ],
                    dim=-1,
                  )
        sin_stk = torch.concat(
                    [
                        torch.sin((2**l)*torch.pi*x) \
                        for l in range(self.L)
                    ],
                    dim=-1,
                  )
        return torch.cat([cos_stk, sin_stk], dim=-1)

class NeRF(nn.Module):
    def __init__(self, depth=8, hidden_dim=256, skips=[4], L_pos=10, L_view=4):
        '''
        Original NeRF Model
        '''
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.L_pos = L_pos
        self.L_view = L_view
        self.pos_dim = 3 * 2 * L_pos
        self.view_dim = 3 * 2 * L_view
        self.skips = skips

        # Positional Encoding
        self.PE_pos = PositionalEncoding(L=L_pos)
        self.PE_view = PositionalEncoding(L=L_view)

        # MLP
        self.linears = nn.ModuleList(
            [nn.Linear(self.pos_dim, hidden_dim)] + \
            [
                nn.Linear(hidden_dim, hidden_dim) if idx not in self.skips \
                else nn.Linear(hidden_dim+self.pos_dim, hidden_dim) \
                for idx in range(depth - 1)
            ]
        )
        self.feat_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dense_linear = nn.Linear(hidden_dim, 1)
        self.view_linears = nn.ModuleList([
            nn.Linear(hidden_dim+self.view_dim, hidden_dim//2)
        ])
        self.rgb_linear = nn.Linear(hidden_dim//2, 3)

    def forward(self, poses, views):
        '''
        input:
            poses: Normalized coordinate values, (N, 3)
            views: Cartesian viewing direction unit vector, (N, 3)
        output:
            denses: volumen density (N, 1)
            rgbs: RGB (N, 3)
        '''
        pe_poses = self.PE_pos(poses)
        pe_views = self.PE_view(views)

        hiddens = pe_poses
        for idx, block in enumerate(self.linears):
            hiddens = self.linears[idx](hiddens)
            hiddens = F.relu(hiddens)
            if idx in self.skips:
                hiddens = torch.cat([pe_poses, hiddens], -1)
        
        denses = self.dense_linear(hiddens)

        feats = self.feat_linear(hiddens)
        hiddens = torch.cat([pe_views, hiddens], -1)
        for idx, block in enumerate(self.view_linears):
            hiddens = self.view_linears[idx](hiddens)
            hiddens = F.relu(hiddens)
        rgbs = self.rgb_linear(hiddens)
        return denses, rgbs

# Style Transfer
class VGGEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        if pretrained:
            features = torchvision.models.vgg19(
                        weights=torchvision.models.VGG19_Weights.DEFAULT
                       ).features
        else:
            features = torchvision.models.vgg19().features
        self.blocks = nn.ModuleList([
                        features[:2],
                        features[2:7],
                        features[7:12],
                        features[12:21]
                      ])
        self.freeze()

    def freeze(self):
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
    
    def forward(self, x, return_last=True):
        '''
        input:
            x: images
            return_last: return last layer of feature
        output:
            output: features (N, C, H, W)
            output: features [(N, C, H, W), ...]
        '''
        if return_last:
            for block in self.blocks:
                x = block(x)
            return x
        else:
            output = [x]
            for block in self.blocks:
                output.append(block(output[-1]))
            return output[1:]

class VGGDecoder(nn.Module):
    def __init__(self, interpolate_mode='bilinear'):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        block1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        block2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )
        block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )
        block4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1, padding_mode="reflect"),
        )
        self.blocks = nn.ModuleList([block1, block2, block3, block4])
    
    def forward(self, f):
        '''
        input:
            f: features
        output:
            output: images
        '''
        output = f
        for idx, block in enumerate(self.blocks):
            output = block(output)
            if idx < len(self.blocks) - 1:
                output = F.interpolate(
                            output,
                            scale_factor=2,
                            mode=self.interpolate_mode
                         )
        return output

class AdaIN(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def cal_mean_std(self, feats):
        assert(len(feats.shape) == 4)
        n, c, _, _ = feats.shape
        feats = feats.flatten(start_dim=2)
        means = torch.mean(feats, dim=-1).view(n, c, 1, 1)
        stds = torch.std(feats, dim=-1).view(n, c, 1, 1) + self.eps
        return means, stds
    
    def forward(self, c_feats, s_feats):
        '''
        input:
            c_feats: content features (N, C, H, W)
            s_feats: style features (N, C, H, W)
        output:
            norm_feats: features after AdaIN (N, C, H, W)
        '''
        c_means, c_stds = self.cal_mean_std(c_feats)
        s_means, s_stds = self.cal_mean_std(s_feats)
        norm_feats = s_stds * (c_feats - c_means) / c_stds + s_means
        return norm_feats

class StyleTransferAdaIN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.enc = VGGEncoder(pretrained=pretrained)
        self.dec = VGGDecoder()
        self.adain = AdaIN()

    def forward(self, c_images, s_images, return_hidden=False):
        '''
        input:
            c_images: content images (N, C, H, W)
            s_images: style images (N, C, H, W)
        output:
            c_feats: content features (N, C, H, W)
            s_feats: style features (N, C, H, W)
            norm_feats: AdaIN features (N, C, H, W)
            output_images: (N, C, H, W)
        '''
        c_feats = self.enc(c_images)
        s_feats = self.enc(s_images)
        norm_feats = self.adain(c_feats, s_feats)
        output_images = self.dec(norm_feats)
        if return_hidden:
            return c_feats, s_feats, norm_feats, output_images
        else:
            return output_images
