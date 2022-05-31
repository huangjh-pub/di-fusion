# DI-Fusion

This repository contains the implementation of our CVPR 2021 paper: DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors.

[Jiahui Huang](https://cg.cs.tsinghua.edu.cn/people/~huangjh/), [Shi-Sheng Huang](https://cg.cs.tsinghua.edu.cn/people/~shisheng/), Haoxuan Song, [Shi-Min Hu](https://cg.cs.tsinghua.edu.cn/shimin.htm)

## Introduction

DI-Fusion (Deep Implicit Fusion) is a novel online 3D reconstruction system based on RGB-D streams. It simultaneously localizes the camera and builds a local implicit map parametrized by a deep network. Please refer to our [technical report](http://arxiv.org/abs/2012.05551) for more details.

## Implementations

We provide two implementations based on Jittor and Pytorch (the latter is also available on the `public` branch).
Please refer to the corresponding folders `jittor/` and `pytorch/` for specific build and running instructions.

## Citation

Please consider citing the following work:
```bibtex
@inproceedings{huang2021difusion,
  title={DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors},
  author={Huang, Jiahui and Huang, Shi-Sheng and Song, Haoxuan and Hu, Shi-Min},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
