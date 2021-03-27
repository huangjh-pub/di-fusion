# Layer class warpped around pytorch layer with customized name and combinations
import torch
import torch.nn as nn


class _INBase(nn.Sequential):
    def __init__(self, in_size, instance_norm=None, name=""):
        super(_INBase, self).__init__()
        self.add_module(name + "in", instance_norm(in_size))
        # Instance norm do not have learnable affine parameters.
        # They only have running metrics, which do need initialization


class InstanceNorm1d(_INBase):
    def __init__(self, in_size, name=""):
        super(InstanceNorm1d, self).__init__(in_size, instance_norm=nn.InstanceNorm1d, name=name)


class InstanceNorm2d(_INBase):
    def __init__(self, in_size, name=""):
        super(InstanceNorm2d, self).__init__(in_size, instance_norm=nn.InstanceNorm2d, name=name)


class InstanceNorm3d(_INBase):
    def __init__(self, in_size, name=""):
        super(InstanceNorm3d, self).__init__(in_size, instance_norm=nn.InstanceNorm3d, name=name)


class GroupNorm(nn.Sequential):
    def __init__(self, in_size, num_groups, name=""):
        super(GroupNorm, self).__init__()
        self.add_module(name + "gn", nn.GroupNorm(num_groups, in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0.0)


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super(_BNBase, self).__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):
    def __init__(self, in_size, name=""):
        super(BatchNorm1d, self).__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size, name=""):
        super(BatchNorm2d, self).__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):
    def __init__(self, in_size, name=""):
        super(BatchNorm3d, self).__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


def get_norm_layer(layer_def, dimension, **kwargs):
    # Layer def is given by user.
    # kwargs are other necessary parameters needed.
    if layer_def is None:
        return nn.Identity()
    class_name = layer_def["class"]
    kwargs.update(layer_def)
    del kwargs["class"]
    return {
        "InstanceNorm": [InstanceNorm1d, InstanceNorm2d, InstanceNorm3d][dimension - 1],
        "GroupNorm": GroupNorm,
        "BatchNorm": [BatchNorm1d, BatchNorm2d, BatchNorm3d][dimension - 1]
    }[class_name](**kwargs)


class _ConvBase(nn.Sequential):

    def __init__(self, in_size, out_size, kernel_size, stride, padding, dilation,
                 activation, bn, bn_dim, init, conv=None,
                 bias=True, preact=False, name=""):
        super(_ConvBase, self).__init__()

        bias = bias and (bn is None)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn is not None:
            if not preact:
                bn_unit = get_norm_layer(bn, bn_dim, in_size=out_size)
            else:
                bn_unit = get_norm_layer(bn, bn_dim, in_size=in_size)

        if preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv1d(_ConvBase):
    def __init__(self, in_size, out_size, kernel_size=1, stride=1, padding=0, dilation=1,
                 activation=nn.ReLU(inplace=True), bn=None, init=nn.init.kaiming_normal_,
                 bias=True, preact=False, name=""):
        super(Conv1d, self).__init__(
            in_size, out_size, kernel_size, stride, padding, dilation,
            activation, bn, 1,
            init, conv=nn.Conv1d,
            bias=bias, preact=preact, name=name)


class Conv2d(_ConvBase):
    def __init__(self, in_size, out_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 activation=nn.ReLU(inplace=True), bn=None, init=nn.init.kaiming_normal_,
                 bias=True, preact=False, name=""):
        super(Conv2d, self).__init__(
            in_size, out_size, kernel_size, stride, padding, dilation,
            activation, bn, 2,
            init, conv=nn.Conv2d,
            bias=bias, preact=preact, name=name)


class Conv3d(_ConvBase):
    def __init__(self, in_size, out_size, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 activation=nn.ReLU(inplace=True), bn=None, init=nn.init.kaiming_normal_,
                 bias=True, preact=False, name=""):
        super(Conv3d, self).__init__(
            in_size, out_size, kernel_size, stride, padding, dilation,
            activation, bn, 3,
            init, conv=nn.Conv3d,
            bias=bias, preact=preact, name=name)


class FC(nn.Sequential):

    def __init__(self,
                 in_size,
                 out_size,
                 activation=nn.ReLU(inplace=True),
                 bn=None,
                 init=None,
                 preact=False,
                 name=""):
        super(FC, self).__init__()

        fc = nn.Linear(in_size, out_size, bias=bn is None)
        if init is not None:
            init(fc.weight)
        if bn is None:
            nn.init.constant_(fc.bias, 0)

        if bn is not None:
            if not preact:
                bn_unit = get_norm_layer(bn, 1, in_size=out_size)
            else:
                bn_unit = get_norm_layer(bn, 1, in_size=in_size)

        if preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class SharedMLP(nn.Sequential):
    def __init__(self, args,
                 bn=None, activation=nn.ReLU(inplace=True), last_act=True, name=""):
        super(SharedMLP, self).__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv1d(
                    args[i],
                    args[i + 1],
                    bn=bn if (last_act or (i != len(args) - 2)) else None,
                    activation=activation if (last_act or (i != len(args) - 2)) else None
                ))


class MLP(nn.Sequential):
    def __init__(self, args,
                 bn=None, activation=nn.ReLU(inplace=True), last_act=True, name=""):
        super(MLP, self).__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                FC(
                    args[i],
                    args[i + 1],
                    bn=bn if (last_act or (i != len(args) - 2)) else None,
                    activation=activation if (last_act or (i != len(args) - 2)) else None
                ))


class Seq(nn.Sequential):

    def __init__(self, input_channels):
        super(Seq, self).__init__()
        self.count = 0
        self.current_channels = input_channels

    def conv1d(self,
               out_size,
               kernel_size=1,
               stride=1,
               padding=0,
               dilation=1,
               activation=nn.ReLU(inplace=True),
               bn=None,
               init=nn.init.kaiming_normal_,
               bias=True,
               preact=False,
               name=""):

        self.add_module(
            str(self.count),
            Conv1d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def conv2d(self,
               out_size,
               kernel_size=(1, 1),
               stride=(1, 1),
               padding=(0, 0),
               dilation=(1, 1),
               activation=nn.ReLU(inplace=True),
               bn=None,
               init=nn.init.kaiming_normal_,
               bias=True,
               preact=False,
               name=""):

        self.add_module(
            str(self.count),
            Conv2d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def conv3d(self,
               out_size,
               kernel_size=(1, 1, 1),
               stride=(1, 1, 1),
               padding=(0, 0, 0),
               dilation=(1, 1, 1),
               activation=nn.ReLU(inplace=True),
               bn=None,
               init=nn.init.kaiming_normal_,
               bias=True,
               preact=False,
               name=""):
        self.add_module(
            str(self.count),
            Conv3d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def fc(self,
           out_size,
           activation=nn.ReLU(inplace=True),
           bn=None,
           init=None,
           preact=False,
           name=""):

        self.add_module(
            str(self.count),
            FC(self.current_channels,
               out_size,
               activation=activation,
               bn=bn,
               init=init,
               preact=preact,
               name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def dropout(self, p=0.5):
        # type: (Seq, float) -> Seq

        self.add_module(str(self.count), nn.Dropout(p=0.5))
        self.count += 1

        return self

    def maxpool2d(self,
                  kernel_size,
                  stride=None,
                  padding=0,
                  dilation=1,
                  return_indices=False,
                  ceil_mode=False):
        self.add_module(
            str(self.count),
            nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode))
        self.count += 1

        return self
