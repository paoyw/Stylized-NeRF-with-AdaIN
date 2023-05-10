import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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
