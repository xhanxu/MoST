import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn import init
from einops import rearrange

# Import dependencies 
from knn_cuda import KNN
from .rmsnorm import *

class K_Rectify(nn.Module):    
    def __init__(self, group_size, h_size=384, eps=0.05, out=False):
        super().__init__()
        self.group_size = group_size
        self.knn = KNN(k=self.group_size + 1, transpose_mode=True)
        self.knorm = rmsnorm(dim=h_size)
        self.ln = nn.LayerNorm(h_size)
        self.eps = eps
        self.out = out

    def _idw(self, distance, feat):
        weights = 1.0 / (distance + self.eps)
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        weights = weights.unsqueeze(-1)
        return torch.sum(weights * feat, dim=2)

    def _extract_cls(self, feat, rf):
        if feat.shape[1] != 128:
            assert feat.shape[1] > 128
            t = feat.shape[1] - 128
            cls_token = feat[:, :t, :]
            feat = feat[:, t:, :]
            rf = rf[t:]
            return feat, rf, cls_token, True
        return feat, rf, None, False

    def _validate_shape(self, feat):
        batch_size, num_points, C = feat.shape
        assert num_points == 128, f"Expected 128 points, got {num_points}"
        return batch_size, num_points, C

    def _compute_c(self, rf):
        return adabias(rf)

    def _gather(self, feat, idx):
        batch_size, num_points, C = feat.shape
        nf = feat.contiguous().view(-1, C)[idx, :]
        assert nf.shape[-1] == feat.shape[-1]
        return nf.view(batch_size, num_points, self.group_size, feat.shape[-1]).contiguous()

    def _centering(self, nf, f, c):
        nf = nf - f.unsqueeze(2)
        return nf, c + f

    def _apply_smooth(self, distance, nf):
        return self._idw(distance, nf)

    def _apply_res(self, f, sf):
        return f + self.knorm(sf)

    def _restore_cls(self, feat, cls_token, cls_flag):
        if cls_flag:
            return torch.cat([cls_token, feat], dim=1)
        return feat

    def forward(self, f, distance, idx, rf):
        f, rf, cls_token, cls_flag = self._extract_cls(f, rf)
        bs, n_pts, C = self._validate_shape(f)
        c = self._compute_c(rf)
        nf = self._gather(f, idx)
        nf, f = self._centering(nf, f, c)
        sf = self._apply_smooth(distance, nf)
        output = self._apply_res(f, sf)
        output = self._restore_cls(output, cls_token, cls_flag)

        return output
