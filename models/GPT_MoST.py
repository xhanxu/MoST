import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

from knn_cuda import KNN
from utils.misc import *

from .structured.modules.point_monarch import PointMonarch

class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: int = 8, lora_dropout: float = 0.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scaling = alpha / rank
        self._init_dropout(lora_dropout)
        self.reset_parameters()
    
    def _init_dropout(self, lora_dropout):
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
    
    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if hasattr(self, 'lora_dropout'):
            x = self.lora_dropout(x)
        
        lora_out = x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return lora_out * self.scaling

    
class K_Smooth_Matching(nn.Module):
    def __init__(self, group_size: int):

        super().__init__()
        self.group_size = group_size
        self.knn = KNN(k=self.group_size + 1, transpose_mode=True)

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_points, _ = xyz.shape

        distance, neighbor_idx = self.knn(xyz, xyz)
        
        assert neighbor_idx.size(1) == num_points, f"Expected {num_points} points, got {neighbor_idx.size(1)}"
        assert neighbor_idx.size(2) == self.group_size + 1, f"Expected {self.group_size + 1} neighbors, got {neighbor_idx.size(2)}"
        
        distance = distance[:, :, 1:]
        neighbor_idx = neighbor_idx[:, :, 1:]
        
        batch_offset = torch.arange(
            0, batch_size, 
            device=xyz.device, 
            dtype=neighbor_idx.dtype
        ).view(-1, 1, 1) * num_points
        
        global_idx = neighbor_idx + batch_offset
        flattened_idx = global_idx.view(-1)
        
        return distance, flattened_idx

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, monarch_nblocks=8):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # PointMonarch
        self.point_monarch = PointMonarch(
            embed_dim=embed_dim, 
            monarch_nblocks=monarch_nblocks,
            group_size=4
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, attn_mask, distance, idx):
        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        r = self.point_monarch(x, distance, idx, a)
        x = x + r
        m = self.ln_2(x)
        m = self.mlp(m)
        x = x + m
        return x 

class GPT_extractor(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        num_layers, 
        num_classes, 
        trans_dim, 
        group_size, 
        pretrained=False, 
        monarch_nblocks=8, 
        fusion_layer=3
    ):
        super(GPT_extractor, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size
        self.fusion_layer = fusion_layer

        self.sos = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList([
            Block(embed_dim, num_heads, monarch_nblocks=monarch_nblocks)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        self.k_smooth_match = K_Smooth_Matching(group_size=4)

        if not pretrained:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
            self.cls_norm = nn.LayerNorm(self.trans_dim)
            self.cls_token_norm = nn.LayerNorm(self.trans_dim)

    def token_sample(self, x):
        result = None
        for i in range(len(x)):
            pooled_features = x[i][2:].max(0)[0] + x[i][2:].mean(0)
            weighted_features = 2 ** i * pooled_features
            
            if i == 0:
                result = weighted_features
            else:
                result += weighted_features
        
        return result

    def forward(self, center, h, pos, attn_mask, classify=False):
        batch, length, C = h.shape

        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)

        sos_token = torch.ones(1, batch, self.embed_dim, device=h.device) * self.sos
        if not classify:
            h = torch.cat([sos_token, h[:-1, :, :]], axis=0)
        else:
            h = torch.cat([sos_token, h], axis=0)

        distance, idx = self.k_smooth_match(center)

        feature_list = []
        fetch_idx = list(range(23, 0, -int(24 / self.fusion_layer)))
        
        for i, layer in enumerate(self.layers):
            h = layer(h + pos, attn_mask, distance, idx)
            if i in fetch_idx:
                feature_list.append(h)

        h_last = self.ln_f(feature_list[-1])
        encoded_points = h_last.transpose(0, 1)
        
        if not classify:
            return encoded_points

        h_last = h_last.transpose(0, 1)
        h_last = self.cls_norm(h_last)
        h_sample = self.token_sample(feature_list)
        h_sample = self.cls_token_norm(h_sample)
        
        concat_features = torch.cat([
            h_last[:, 1], 
            h_last[:, 2:].max(1)[0] + h_sample
        ], dim=-1)
        
        classification_result = self.cls_head_finetune(concat_features)
        return classification_result, encoded_points, distance, idx


class GPT_generator(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        num_layers, 
        trans_dim, 
        group_size, 
        monarch_nblocks=8
    ):
        super(GPT_generator, self).__init__()

        self.embed_dim = embed_dim
        self.trans_dim = trans_dim
        self.group_size = group_size

        self.sos = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.layers = nn.ModuleList([
            Block(embed_dim, num_heads, monarch_nblocks=monarch_nblocks)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

    def forward(self, h, pos, attn_mask, distance, idx):
        batch, length, C = h.shape

        h = h.transpose(0, 1)
        pos = pos.transpose(0, 1)

        for layer in self.layers:
            h = layer(h + pos, attn_mask, distance, idx)

        h = self.ln_f(h)

        rebuild_points = self.increase_dim(h.transpose(1, 2)).transpose(
            1, 2).transpose(0, 1).reshape(batch * length, -1, 3)

        return rebuild_points
