# DI-Fusion: 基于深度先验的在线隐式三维重建
本仓库是[此文章](https://github.com/huangjh-pub/di-fusion)提出的用于隐式三维重建的网络部分的[计图](https://cg.cs.tsinghua.edu.cn/jittor/)实现，作者是：[黄家晖](https://cg.cs.tsinghua.edu.cn/people/~huangjh/zh)、[黄石生](https://cg.cs.tsinghua.edu.cn/people/~shisheng/)、宋浩轩和[胡事民](https://cg.cs.tsinghua.edu.cn/shimin.htm)

- 计图是由清华大学计算机系 [图形学实验室](https://cg.cs.tsinghua.edu.cn/) 推出的一个完全基于动态编译、内部使用创新的元算子和统一计算图的深度学习框架。

- DI-Fusion是一个基于RGBD输入的在线三维重建系统。它的相机定位追踪模块以及地图表示完全基于由深度神经网络建模的局部隐式表示。请进一步参考我们的[ArXiv报告](http://arxiv.org/abs/2012.05551)和[视频](https://youtu.be/yxkIQFXQ6rw)。
- PyTorch version **available [here](https://github.com/huangjh-pub/di-fusion).**

## 网络训练

### 训练数据生成

首先，请编译我们的CUDA点云采样器：

```bash
cd sampler_cuda
mkdir build; cd build
cmake ..
make -j
```

编译成功之后，会在`sampler_cuda/bin/`文件夹下，生成名为`PreprocessMeshCUDA`的可执行文件，接着运行：

```bash
python data_generator.py data-shapenet.yaml --nproc 4
```

即可生成用于训练的数据。

### 网络训练

当完成了训练数据的生成之后，请运行：

```bash
python train.py train.yaml
```

即可开始训练，如果您在上一步更改了数据存放的位置，可能需要修改`train.yaml`中对应的路径，让程序正确的找到数据集。

### 速度对比

计图框架采用了先进的元算子融合以及统一计算图技术，这使得执行效率大大提高，下表对比了使用计图的版本和使用PyTorch的版本的训练速度，训练同一个epoch**计图所用的时间仅有PyTorch的三分之一**，原需要训练1-2天的模型，可以在半天内取得较好的收敛效果。

|                     | PyTorch | PyTorch JIT | 计图 |
| ------------------- | ------- | ----------- | ---- |
| 每秒训练步数 (it/s) | 13      | 14          | 39   |

## 运行 (beta)

如果需要运行完整的DI-Fusion系统，首先需要进行简单的权值格式转换：

```bash
python convert.py
```

假设本仓库的路径是`<DIR>`，那么上述权值转换程序将输出`<DIR>/model_300.pth.tar`和`<DIR>/encoder_300.pth.tar`两个文件。

接着，执行如下命令，拷贝官方实现的代码：

```bash
git clone https://github.com/huangjh-pub/di-fusion.git
cd di-fusion
cp <DIR>/model_300.pth.tar ./ckpt/default/
cp <DIR>/encoder_300.pth.tar ./ckpt/default/
```

执行完成之后，请依照[这里](https://github.com/huangjh-pub/di-fusion#running)的做法继续之后的操作步骤。

## 引用

欢迎您引用我们的工作，蟹蟹：

```
@inproceedings{huang2021difusion,
  title={DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors},
  author={Huang, Jiahui and Huang, Shi-Sheng and Song, Haoxuan and Hu, Shi-Min},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

