#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, latent_size, dims, dropout=None, dropout_prob=0.0, norm_layers=(), latent_in=(), weight_norm=False):
        """
        :param latent_size: Size of the latent vector
        :param dims:        Intermediate network neurons
        :param dropout:
        :param dropout_prob:
        :param latent_in:   From which layer to re-feed in the latent_size+3 input vector
        :param weight_norm: Whether to use weight normalization
        :param norm_layers: Layers to append normalization (Either WeightNorm or LayerNorm depend of weight_norm var)
        """
        super().__init__()

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.weight_norm = weight_norm

        for layer in range(self.num_layers - 1):
            # a linear layer with input `dims[layer]` and output `dims[layer + 1]`.
            # If the next layer is going to take latent vec, we reduce output of this layer.
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]

            if weight_norm and layer in self.norm_layers:
                setattr(self, "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (not weight_norm) and self.norm_layers is not None and layer in self.norm_layers:
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.uncertainty_layer = nn.Linear(dims[-2], 1)

        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        x = input
        std = None

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)

            # For the last layer, also branch out to output uncertainty.
            if layer == self.num_layers - 2:
                std = self.uncertainty_layer(x)
                # std = 0.1 + 0.9 * F.softplus(std)
                std = 0.05 + 0.5 * F.softplus(std)

            x = lin(x)

            # For all layers other than the last layer.
            if layer < self.num_layers - 2:
                if (
                        self.norm_layers is not None
                        and layer in self.norm_layers
                        and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.th(x)

        return x, std
