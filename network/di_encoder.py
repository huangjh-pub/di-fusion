import torch
import torch.nn as nn
from utils.pt_util import SharedMLP


class Model(nn.Module):
    def __init__(self, bn, latent_size, per_point_feat, mode='cnp'):
        super().__init__()
        assert mode in ['train', 'cnp']
        self.mode = mode
        self.aggr_mode = 'mean'
        self.mlp = SharedMLP(per_point_feat + [latent_size], bn=bn, last_act=False)

    def forward(self, x):
        if self.mode == 'train':
            # B x N x 3 -> B x latent_size
            x = x.transpose(-1, -2)
            x = self.mlp(x)     # (B, L, N)
            if self.aggr_mode == 'max':
                r, _ = torch.max(x, dim=-1)
            elif self.aggr_mode == 'mean':
                r = torch.mean(x, dim=-1)
            else:
                raise NotImplementedError
            return r
        elif self.mode == 'cnp':
            # B x 3 -> B x latent_size
            x = x.unsqueeze(-1)     # (B, 3, 1)
            x = self.mlp(x)         # (B, L, 1)
            return x.squeeze(-1)
        else:
            raise NotImplementedError
