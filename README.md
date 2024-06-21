# Stylize NeRF with AdaIN 

[![arXiv](https://img.shields.io/badge/arXiv-2406.04960-b31b1b.svg)](https://arxiv.org/abs/2406.04960)

[![slide](https://img.shields.io/badge/google%20slide-introduction-ffba00.svg)](https://docs.google.com/presentation/d/e/2PACX-1vS8BNl5ONMOmT4AqOY0WVw8T3ZR-cLWvtVA3hgSiAsTg46B0YKnTGRVgDEp_IZHtNNeEHC_VDWimkUv/pub?start=false&loop=false&delayms=3000)

## Visual results
![Multi-NeRF flower 0](./img/multi-flower1.gif)

![Multi-NeRF flower 1](./img/multi-flower2.png)

![Multi-NeRF room](./img/multi-room.png)

![MultiNeRF leave](./img/multi-leaves.png)

![Interpolate dinosaur](./img/interpolate-dinosaur.gif)

![Interpolate horn](./img/interpolate-horn.png)

## AdaIN results

![AdaIN result 0](./img/flower0.jpg)

![AdaIN result 1](./img/flower1.jpg)

![AdaIN result 2](./img/flower2.jpg)

![AdaIN result 3](./img/trex0.jpg)

![AdaIN result 4](./img/trex1.jpg)

![AdaIN result 5](./img/trex2.jpg)

![AdaIN result 6](./img/lego0.png)

![AdaIN result 7](./img/lego1.png)

![AdaIN result 8](./img/drums0.png)

![AdaIN result 9](./img/drums1.png)

![AdaIN result 10](./img/drums2.png)

# NeRF results

![NeRF result 0](./img/flower_test_spiral_200000_rgb.gif)

# Directly Style Transfer by AdaIN on NeRF

![AdaIN on NeRF result 0](./img/lego-EnCampoGris.gif)

![AdaIN on NeRF result 1](./img/ship-FlowerFishAndFruit.gif)

# Train NeRF on Style-Transfered Images by AdaIN

![NeRF on AdaIN result 0](./img/blender_paper_lego-EnCampoGris_spiral_150000_rgb.gif)

![NeRF on AdaIN result 1](./img/ship-FlowerFishandFruit-apng.png)

![NeRF on AdaIN result 2](./img/trex_test-Bacchante_spiral_200000_rgb.png)

![NeRF on AdaIN result 3](./img/horns_test-TheStarryNight_spiral_200000_rgb.png)

## Introduction and More Results

[Stylize NeRF with AdaIN](https://docs.google.com/presentation/d/e/2PACX-1vS8BNl5ONMOmT4AqOY0WVw8T3ZR-cLWvtVA3hgSiAsTg46B0YKnTGRVgDEp_IZHtNNeEHC_VDWimkUv/pub?start=false&loop=false&delayms=3000)


## Citation
The original paper of AdaIN
```
@inproceedings{huang2017adain,
  title={Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization},
  author={Huang, Xun and Belongie, Serge},
  booktitle={ICCV},
  year={2017}
}
```

The original paper of NeRF
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

pytorch-implementation for NeRF
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
