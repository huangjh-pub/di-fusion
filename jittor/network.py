import math
import jittor as jt
import jittor.nn as nn


SHAPE_MULT = 1


class WNLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(WNLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = nn.init.invariant_uniform((out_features, in_features), "float32")
        self.weight_g = jt.norm(self.weight_v, k=2, dim=1, keepdim=True)
        bound = 1.0 / math.sqrt(in_features)
        self.bias = nn.init.uniform((out_features,), "float32", -bound, bound) if bias else None

    def execute(self, x):
        weight = self.weight_g * (self.weight_v / jt.norm(self.weight_v, k=2, dim=1, keepdim=True))
        x = nn.matmul_transpose(x, weight)
        if self.bias is not None:
            return x + self.bias
        return x


class DIDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = WNLinear(32, 128 * SHAPE_MULT)
        self.lin1 = WNLinear(128 * SHAPE_MULT, 128 * SHAPE_MULT)
        self.lin2 = WNLinear(128 * SHAPE_MULT, 128 * SHAPE_MULT - 32)
        self.lin3 = WNLinear(128 * SHAPE_MULT, 128 * SHAPE_MULT)
        self.lin4 = WNLinear(128 * SHAPE_MULT, 1)
        self.uncertainty_layer = nn.Linear(128 * SHAPE_MULT, 1)
        self.relu = nn.ReLU()
        self.dropout = [0, 1, 2, 3, 4, 5]
        self.th = nn.Tanh()

    def execute(self, ipt):
        x = self.lin0(ipt)
        x = self.relu(x)
        x = nn.dropout(x, p=0.2, is_train=True)

        x = self.lin1(x)
        x = self.relu(x)
        x = nn.dropout(x, p=0.2, is_train=True)

        x = self.lin2(x)
        x = self.relu(x)
        x = nn.dropout(x, p=0.2, is_train=True)

        x = jt.contrib.concat([x, ipt], 1)
        x = self.lin3(x)
        x = self.relu(x)
        x = nn.dropout(x, p=0.2, is_train=True)

        std = self.uncertainty_layer(x)
        std = 0.05 + 0.5 * nn.softplus(std)
        x = self.lin4(x)
        x = self.th(x)

        return x, std


class DIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(6, 32 * SHAPE_MULT, kernel_size=1, bias=False), nn.BatchNorm1d(32 * SHAPE_MULT), nn.ReLU(),
            nn.Conv1d(32 * SHAPE_MULT, 64 * SHAPE_MULT, kernel_size=1, bias=False), nn.BatchNorm1d(64 * SHAPE_MULT), nn.ReLU(),
            nn.Conv1d(64 * SHAPE_MULT, 256 * SHAPE_MULT, kernel_size=1, bias=False), nn.BatchNorm1d(256 * SHAPE_MULT), nn.ReLU(),
            nn.Conv1d(256 * SHAPE_MULT, 29, kernel_size=1, bias=True)
        )

    def execute(self, x):
        x = x.transpose([0, 2, 1])
        x = self.mlp(x)  # (B, L, N)
        r = jt.mean(x, dim=-1)
        return r
