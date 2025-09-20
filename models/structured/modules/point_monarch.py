import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn import init
from einops import rearrange

# Import dependencies 
from knn_cuda import KNN
from .rmsnorm import *
from .local_smooth import K_Rectify
from .structured_linear import StructuredLinear
from .monarch_linear import MonarchLinear
from .blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from .blockdiag_butterflt_einsum import blockdiag_butterfly_project_einsum_rank


class PointMonarch(nn.Module):
    """
    K_Rectify → MonarchLinear → K_Rectify
    """
    
    def __init__(self, embed_dim: int, monarch_nblocks: int = 8, group_size: int = 4, eps: float = 0.05):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.monarch_nblocks = monarch_nblocks
        self.group_size = group_size
        self.eps = eps
        
        self._init_smoothers()
        self._init_linear()
    
    def _init_smoothers(self):
        self.spatial_smooth_in = K_Rectify(
            group_size=self.group_size, 
            h_size=self.embed_dim, 
            eps=self.eps, 
            out=False
        )
        
        self.spatial_smooth_out = K_Rectify(
            group_size=self.group_size, 
            h_size=self.embed_dim, 
            eps=self.eps, 
            out=True
        )
    
    def _init_linear(self):
        self.monarch_linear = MonarchLinear(
            self.embed_dim, 
            self.embed_dim, 
            nblocks=self.monarch_nblocks, 
            bias=True
        )
    
    def _prepare_in(self, x: torch.Tensor, rf: torch.Tensor):
        return x.transpose(1, 0).contiguous(), rf.contiguous()
    
    def _out_transform(self, x: torch.Tensor, rf: torch.Tensor):
        return x.transpose(1, 0).contiguous() + rf
    
    def _smooth_in(self, x: torch.Tensor, distance: torch.Tensor, idx: torch.Tensor, rf: torch.Tensor):
        return self.spatial_smooth_in(x, distance, idx, rf)
    
    def _linear(self, x: torch.Tensor, rf: torch.Tensor):
        return self.monarch_linear(x)

    def _smooth_out(self, x: torch.Tensor, distance: torch.Tensor, idx: torch.Tensor, rf: torch.Tensor):
        return self.spatial_smooth_out(x, distance, idx, rf)
    
    def forward(self, x: torch.Tensor, distance: torch.Tensor, idx: torch.Tensor, rf: torch.Tensor):
        x_spatial, refeat_spatial = self._prepare_in(x, rf)

        smoothed_input = self._smooth_in(x_spatial, distance, idx, refeat_spatial)

        transformed = self._linear(smoothed_input, rf)

        smoothed_output = self._smooth_out(transformed, distance, idx, refeat_spatial)

        final_output = self._out_transform(smoothed_output, rf)
        return final_output
    
    @property
    def parameter_efficiency(self):
        return self.monarch_linear.saving

    def reset_parameters(self):
        self.monarch_linear.reset_parameters()
    
    def get_component_info(self):
        return {
            'spatial_smooth_in': {
                'group_size': self.spatial_smooth_in.group_size,
                'eps': self.spatial_smooth_in.eps,
                'type': 'K_Rectify'
            },
            'monarch_linear': {
                'nblocks': self.monarch_linear.nblocks,
                'saving': self.monarch_linear.saving,
                'type': 'MonarchLinear'
            },
            'spatial_smooth_out': {
                'group_size': self.spatial_smooth_out.group_size,
                'eps': self.spatial_smooth_out.eps,
                'type': 'K_Rectify'
            }
        }
    
    def __repr__(self):
        return (f"PointMonarch(\n"
                f"  embed_dim={self.embed_dim},\n"
                f"  monarch_nblocks={self.monarch_nblocks},\n"
                f"  group_size={self.group_size},\n"
                f"  eps={self.eps},\n"
                f"  parameter_efficiency={self.parameter_efficiency:.3f}\n"
                f")")

