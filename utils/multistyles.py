import torch
import torch.nn as nn
import torch.nn.functional as F
import run_nerf_helpers

class MultistylesNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, use_style_density=False):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.use_style_density = use_style_density
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            if use_style_density:
                self.alpha_linear = nn.Sequential(
                    nn.Linear(W+W, W),
                    nn.ReLU(),
                    nn.Linear(W, 1)
                 )
            else:
                self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Sequential(
                nn.Linear(W, W//2),
                nn.ReLU(),
                nn.Linear(W//2, 3),
            )
        else:
            self.output_linear = nn.Linear(W+W//2, output_ch)

        self.style_rgb_mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, W//2),
        )
        if use_style_density:
            self.style_density_mlp = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, W)
            )
        self.freeze()

    def freeze(self,):
        for param in self.pts_linears.parameters():
            param.requires_grad = False

        for param in self.views_linears.parameters():
            param.requires_grad = False

        if self.use_viewdirs:
            for param in self.feature_linear.parameters():
                param.requires_grad = False

        if not self.use_style_density:
            for param in self.alpha_linear.parameters():
                param.requires_grad = False

    def load_pretrained(self, nerf, freeze=True):
        self.pts_linears.load_state_dict(
            nerf.pts_linears.state_dict()
        )
        if freeze:
            for param in self.pts_linears.parameters():
                param.requires_grad = False

        self.views_linears.load_state_dict(
            nerf.views_linears.state_dict()
        )
        if freeze:
            for param in self.views_linears.parameters():
                param.requires_grad = False

        if nerf.use_viewdirs and self.use_viewdirs:
            self.feature_linear.load_state_dict(
                nerf.feature_linear.state_dict()
            )
            if freeze:
                for param in self.feature_linear.parameters():
                    param.requires_grad = False

        if nerf.use_viewdirs and not self.use_style_density:
            self.alpha_linear.load_state_dict(
                nerf.alpha_linear.state_dict(),
            )
            if freeze:
                for param in self.alpha_linear.parameters():
                    param.requires_grad = False

    def forward(self, x, styles):
        style_rgb_h = self.style_rgb_mlp(styles)
        if self.use_style_density:
            style_alpha_h = self.style_density_mlp(styles)

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            if self.use_style_density:
                alpha = self.alpha_linear(
                    torch.cat([h, style_alpha_h], -1)
                )
            else:
                alpha = self.alpha_linear(h)
                
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(
                torch.cat([h, style_rgb_h], -1)
            )

            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(
                torch.cat([h, style_rgb_h], -1)
            )

        return outputs

def nerf2multistylesnerf(args, input_ch, output_ch, skips, input_ch_views, path):
    nerf_save = torch.load(path, map_location='cpu')
    network_fn = run_nerf_helpers.NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs
    )
    network_fn.load_state_dict(nerf_save['network_fn_state_dict'])
    style_network_fn = MultistylesNeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
        use_style_density=args.use_style_density,
    )
    style_network_fn.load_pretrained(network_fn)
    network_fine = run_nerf_helpers.NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs
    )
    network_fine.load_state_dict(nerf_save['network_fine_state_dict'])
    style_network_fine = MultistylesNeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
        use_style_density=args.use_style_density,
    )
    style_network_fine.load_pretrained(network_fine)
    return style_network_fn, style_network_fine
