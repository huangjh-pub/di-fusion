# DI-Fusion

This repository contains the implementation of our CVPR 2021 paper: DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors.

[Jiahui Huang](https://cg.cs.tsinghua.edu.cn/people/~huangjh/), [Shi-Sheng Huang](https://cg.cs.tsinghua.edu.cn/people/~shisheng/), Haoxuan Song, [Shi-Min Hu](https://cg.cs.tsinghua.edu.cn/shimin.htm)

## Introduction

DI-Fusion (Deep Implicit Fusion) is a novel online 3D reconstruction system based on RGB-D streams. It simultaneously localizes the camera and builds a local implicit map parametrized by a deep network. Please refer to our [technical report](http://arxiv.org/abs/2012.05551) and [video]() for more details.

## Training

The network for representing geometry is trained on [ShapeNet](https://shapenet.org/) and learns the relationship between the depth point observations and the local geometry.

### Generate the data

The data generation step is similar to [DeepSDF](https://github.com/facebookresearch/DeepSDF). In this repo we contribute a CUDA-based sampler alternative. To compile, run:

```bash
cd sampler_cuda
mkdir build; cd build
cmake ..
make -j
```

This will gives you a binary named `PreprocessMeshCUDA` under the `sampler_cuda/bin/` directory.

Then run the following script to generate the dataset used for training the encoder-decoder network:

```bash
python data_generator.py configs/data-shapenet.yaml --nproc 4
```

The above yaml file is just one example. You should change the `provider_kwargs.shapenet_path` into your downloaded ShapeNet path. You can also change the provider into `simple_shape` or change the shape categories/samples/scales used for training. In our case, we observe the network can already provide a good fit even if it is trained solely on the chairs.

### Train the network

Once the dataset is generated (it should be fast if you enable multi-processing), run the following script to start training.

```bash
python network_trainer.py configs/train-cnp.yaml
```

The training takes 1-2 days (depends on your GPU and cache speed) and you can open the tensorboard to monitor the training process.

## Running

After obtaining the model you can run our fusion pipeline. A sample recording (ICL NUIM) can be downloaded here ([w/ noise](https://drive.google.com/file/d/1InewwdfQEIe6Qaftqxd6Qhvj3KJVGx2x/view?usp=sharing)) or their [official website](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html).

```bash
python main.py configs/fusion-lr-kt.yaml --vis 1
```

Visualization (the `vis` flag) is not necessary but it gives you a better intuition of what is going on. When the GUI is ready (a white window prompts), press `,` on your keyboard to run step-by-step or press `.` to run continuously. The color of the mesh shows uncertainty of the network estimation.

You can tune the parameters in the configuration yaml file for better tracking and reconstruction performance. Please refer to `configs/fusion-lr-kt.yaml` for the descriptions of the parameters.

## Notes

- The pytorch extensions in `ext/` will be automatically compiled on first run, so expect a delay before the GUI prompts.
- To boost efficiency we update the mesh for visualization in an incremental way, and this may cause glitches/artifacts in the preview mesh, which can be solved by saving and loading the map later. You may also increase `resolution` (default is only 4 to enable interactive GUI) to get a mesh with better quality.

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
