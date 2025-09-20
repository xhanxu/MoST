import math

import torch
import torch.nn as nn
from torch.nn import init

from einops import rearrange

from .structured_linear import StructuredLinear
from .blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from .blockdiag_butterflt_einsum import blockdiag_butterfly_project_einsum_rank


class MonarchLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nblocks = nblocks
        self._compute_block_dimensions()
        self._initialize_block_diagonal_params()
        self.reset_parameters_zero_zero()
    
    def _compute_block_dimensions(self):
        in_blksz = int(math.ceil(self.in_features / self.nblocks))
        out_blksz = int(math.ceil(self.out_features / self.nblocks))
        self.in_blksz = in_blksz
        self.out_blksz = out_blksz
        self.in_features_extended = in_blksz * self.nblocks
        self.out_features_extended = out_blksz * self.nblocks
    
    def _initialize_block_diagonal_params(self):
        if self.in_features_extended < self.out_features_extended:
            self._init_input_dominant_blocks()
        else:
            self._init_output_dominant_blocks()
    
    def _init_input_dominant_blocks(self):
        self.blkdiag1 = nn.Parameter(torch.empty(self.nblocks, self.in_blksz, self.in_blksz))
        self.blkdiag2 = nn.Parameter(torch.empty(self.nblocks, self.out_blksz, self.in_blksz))
    
    def _init_output_dominant_blocks(self):
        self.blkdiag1 = nn.Parameter(torch.empty(self.nblocks, self.out_blksz, self.in_blksz))
        self.blkdiag2 = nn.Parameter(torch.empty(self.nblocks, self.out_blksz, self.out_blksz))
        
    def reset_parameters_zero_zero(self) -> None:
        print("Using blk1:zero blk2:zero init!")
        nn.init.zeros_(self.blkdiag1)
        nn.init.zeros_(self.blkdiag2)

    @property
    def saving(self):
        monarch_params = self.blkdiag1.numel() + self.blkdiag2.numel()
        dense_params = self.in_features * self.out_features
        return monarch_params / dense_params
    
    @property 
    def compression_info(self):
        return {
            'monarch_params': self.blkdiag1.numel() + self.blkdiag2.numel(),
            'dense_params': self.in_features * self.out_features,
            'compression_ratio': self.saving,
            'memory_reduction': f"{(1 - self.saving) * 100:.1f}%"
        }

    def forward_matmul(self, x):
        preprocessed = self.preprocess(x)
        transformed = self._apply_monarch_multiply(preprocessed)
        return self.postprocess(transformed)
    
    def _apply_monarch_multiply(self, x):
        return blockdiag_butterfly_multiply(x, self.blkdiag1, self.blkdiag2)
    
    def __repr__(self):
        return (f"MonarchLinear(\n"
                f"  in_features={self.in_features},\n"
                f"  out_features={self.out_features},\n"
                f"  nblocks={self.nblocks},\n"
                f"  compression_ratio={self.saving:.3f},\n"
                f"  bias={self.bias is not None}\n"
                f")")
