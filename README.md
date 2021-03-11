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
```

This will gives you a binary named `PreprocessMeshCUDA` under the `sampler_cuda/bin/` directory.

Then run the following script to generate the dataset used for training the encoder-decoder network:

```bash
python data_generator.py configs/data/shapenet.yaml
```

The above yaml file is just one example. You should change the `provider_kwargs.shapenet_path` into your downloaded ShapeNet path. You can also change the provider into `` or change the shape categories used for training.

### Train the network

Once the dataset is generated (it should be fast if you enable multi-processing), run the following script to start training.

```bash
python train.py
```

The training takes 1-2 days and you can open the tensorboard to monitor the training process.

## Running

Now you can run the SLAM. Given a yaml file, the system automatically determines the GPUs to run on.

You can tune the parameters in this yaml file for better performance:

- `integrate_interval`:
- `depth_cut_min`, `depth_cut_max`: define the range of depth observations to be cropped. Unit is meter.
- `run_async`: 
- `meshing_interval`:
- `resolution`:
- `mapping`:
  - `bound_min`, `bound_max`:
  - `voxel_size`:
  - `prune_min_vox_obs`:
  - `ignore_count_th`: 
  - `encoder_count_th`:
- `tracking`:
  - `iter_config`: An array defining how the camera pose is optimized. Each element is a dictionary: For example `{"n": 2, "type": [['sdf'], ['rgb', 1]]}`  means to optimize the summation of sdf term and rgb term at the 1st level pyramid for 2 iterations.
  - `sdf.robust_kernel`, `sdf.robust_k`: 

### Example

We provide an example config file to quickly run 

## Notes

- The pytorch extensions in `ext/` will be automatically compiled on first run, so expect a delay before the GUI prompts. If you are experiencing a longer wait than expected, (possibly) please refer to [this issue]().
- Due to unknown reasons the code does not work well with CUDA 11.

## Citation

Please consider citing the following work:
```bibtex
@inproceedings {
}
```
