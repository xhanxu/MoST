import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.GPT_MoST import GPT_extractor, GPT_generator
import math
from models.z_order import *


class Encoder_large(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )
        
        self.second_conv = nn.Sequential(
            nn.Conv1d(2048, 2048, 1),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Conv1d(2048, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        
        feature = torch.cat([
            feature_global.expand(-1, -1, n), feature
        ], dim=1)
        
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        
        return feature_global.reshape(bs, g, self.encoder_channel)


class Encoder_small(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        
        feature = torch.cat([
            feature_global.expand(-1, -1, n), feature
        ], dim=1)
        
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.knn_2 = KNN(k=1, transpose_mode=True)

    def _reshape_distances_batch(self, distances_batch, batch_size):
        return distances_batch.view(
            batch_size, self.num_group, self.num_group
        ).transpose(1, 2).contiguous().view(
            batch_size * self.num_group, self.num_group
        )

    def simplied_morton_sorting(self, xyz, center):
        batch_size, num_points, _ = xyz.shape
        distances_batch = torch.cdist(center, center)
        distances_batch[:, torch.eye(self.num_group).bool()] = float("inf")
        
        idx_base = torch.arange(0, batch_size, device=xyz.device) * self.num_group
        sorted_indices_list = [idx_base]
        
        distances_batch = self._reshape_distances_batch(distances_batch, batch_size)
        distances_batch[idx_base] = float("inf")
        distances_batch = distances_batch.view(
            batch_size, self.num_group, self.num_group
        ).transpose(1, 2).contiguous()
        
        for i in range(self.num_group - 1):
            distances_batch = distances_batch.view(batch_size * self.num_group, self.num_group)
            distances_to_last_batch = distances_batch[sorted_indices_list[-1]]
            closest_point_idx = torch.argmin(distances_to_last_batch, dim=-1) + idx_base
            sorted_indices_list.append(closest_point_idx)
            
            distances_batch = self._reshape_distances_batch(
                distances_batch.view(batch_size, self.num_group, self.num_group), batch_size
            )
            distances_batch[closest_point_idx] = float("inf")
            distances_batch = distances_batch.view(
                batch_size, self.num_group, self.num_group
            ).transpose(1, 2).contiguous()
        
        return torch.stack(sorted_indices_list, dim=-1).view(-1)

    def morton_sorting(self, xyz, center):
        batch_size, num_points, _ = xyz.shape
        all_indices = []
        
        for index in range(batch_size):
            points = center[index]
            z = get_z_values(points.cpu().numpy())
            temp = np.arange(self.num_group)
            z_ind = np.argsort(z[temp])
            all_indices.append(temp[z_ind])
        
        all_indices = torch.tensor(all_indices, device=xyz.device)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1) * self.num_group
        return (all_indices + idx_base).view(-1)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        
        center = misc.fps(xyz, self.num_group)
        _, idx = self.knn(xyz, center)
        
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)
        
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :].view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)

        sorted_indices = self.simplied_morton_sorting(xyz, center)

        neighborhood = neighborhood.view(
            batch_size * self.num_group, self.group_size, 3
        )[sorted_indices, :, :].view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        
        center = center.view(
            batch_size * self.num_group, 3
        )[sorted_indices, :].view(
            batch_size, self.num_group, 3
        ).contiguous()

        return neighborhood, center



class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t,
                                     2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-
                              1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


class GPT_Transformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        
        transformer_config = config.transformer_config
        self.mask_ratio = transformer_config.mask_ratio
        self.trans_dim = transformer_config.trans_dim
        self.depth = transformer_config.depth
        self.decoder_depth = transformer_config.decoder_depth
        self.drop_path_rate = transformer_config.drop_path_rate
        self.num_heads = transformer_config.num_heads
        self.group_size = config.group_size
        print_log(f'[args] {transformer_config}', logger='Transformer')

        self.encoder_dims = transformer_config.encoder_dims
        assert self.encoder_dims in [384, 768, 1024]
        
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            pretrained=True,
        )

        self.generator_blocks = GPT_generator(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.decoder_depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size
        )

        self.keep_attend = 10
        self.num_groups = config.num_group
        self.num_mask = int((self.num_groups - self.keep_attend) * self.mask_ratio)

        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _create_attention_mask(self, group_input_tokens):
        batch_size, seq_len, _ = group_input_tokens.size()
        
        attn_mask = torch.full(
            (seq_len, seq_len), -float("Inf"), 
            device=group_input_tokens.device, 
            dtype=group_input_tokens.dtype
        ).to(torch.bool)

        with torch.no_grad():
            attn_mask = torch.triu(attn_mask, diagonal=1)

            overall_mask = np.hstack([
                np.zeros(self.num_groups - self.keep_attend - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(overall_mask)
            overall_mask = np.hstack([
                np.zeros(self.keep_attend),
                overall_mask,
            ])
            overall_mask = torch.from_numpy(overall_mask).to(torch.bool).to('cuda')
            eye_mask = torch.eye(self.num_groups).to(torch.bool).to('cuda')
            attn_mask = attn_mask | overall_mask.unsqueeze(0) & ~eye_mask

        return attn_mask

    def forward(self, neighborhood, center, noaug=False, classify=False):
        group_input_tokens = self.encoder(neighborhood)
        batch_size, seq_len, C = group_input_tokens.size()

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / relative_norm
        position = torch.cat([center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position)

        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos_absolute = self.pos_embed(center[:, :-1, :])
        pos_absolute = torch.cat([sos_pos, pos_absolute], dim=1)

        attn_mask = self._create_attention_mask(group_input_tokens)

        if not classify:
            encoded_features = self.blocks(
                group_input_tokens, pos_absolute, attn_mask, classify=classify)
            generated_points = self.generator_blocks(
                encoded_features, pos_relative, attn_mask)
            return generated_points
        else:
            print('----error---- This code is detached ----error----')
            logits, generated_points = self.blocks(
                group_input_tokens, pos_absolute, classify=classify)
            return logits, generated_points


@MODELS.register_module()
class PointGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointGPT] ', logger='PointGPT')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.GPT_Transformer = GPT_Transformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.weight_center = config.weight_center

        print_log(
            f'[PointGPT] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger='PointGPT')
        self.group_divider = Group(
            num_group=self.num_group, group_size=self.group_size)

        self.loss = config.loss

        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func_p = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func_p = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl12':
            self.loss_func_p1 = ChamferDistanceL1().cuda()
            self.loss_func_p2 = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
        self.loss_func_c = nn.MSELoss().cuda()

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        B = neighborhood.shape[0]

        generated_points = self.GPT_Transformer(
            neighborhood, center)

        gt_points = neighborhood.reshape(
            B*(self.num_group), self.group_size, 3)
        loss1 = self.loss_func_p1(generated_points, gt_points)
        loss2 = self.loss_func_p2(generated_points, gt_points)

        if vis:  # visualization
            gt_points = gt_points.reshape(
                B, self.num_group, self.group_size, 3)
            gt_points = (gt_points + center.unsqueeze(-2)
                         ).reshape(-1, 3).unsqueeze(0)
            generated_points = generated_points.reshape(
                B, self.num_group, self.group_size, 3) + center.unsqueeze(-2)
            generated_points = generated_points.reshape(-1, 3).unsqueeze(0)

            return generated_points, gt_points, center

        return loss1 + loss2


@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.decoder_depth = config.decoder_depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.monarch_nblocks = config.monarch_nblocks
        self.fusion_layer = config.fusion_layer

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        assert self.encoder_dims in [384, 768, 1024]
        if self.encoder_dims == 384:
            self.encoder = Encoder_small(encoder_channel=self.encoder_dims)
        else:
            self.encoder = Encoder_large(encoder_channel=self.encoder_dims)

        self.pos_embed = PositionEmbeddingCoordsSine(3, self.encoder_dims, 1.0)

        self.blocks = GPT_extractor(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.depth,
            num_classes=config.cls_dim,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            monarch_nblocks=self.monarch_nblocks,
            fusion_layer=self.fusion_layer,
        )

        self.generator_blocks = GPT_generator(
            embed_dim=self.encoder_dims,
            num_heads=self.num_heads,
            num_layers=self.decoder_depth,
            trans_dim=self.trans_dim,
            group_size=self.group_size,
            monarch_nblocks=self.monarch_nblocks,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.sos_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.norm = nn.LayerNorm(self.trans_dim)

        self.build_loss_func()
        
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self, loss_type='cdl12'):
        self.loss_ce = nn.CrossEntropyLoss()
        if loss_type == "cdl1":
            self.loss_func_p = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func_p = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl12':
            self.loss_func_p1 = ChamferDistanceL1().cuda()
            self.loss_func_p2 = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def _load_checkpoint_dict(self, base_ckpt):
        for k in list(base_ckpt.keys()):
            if k.startswith('GPT_Transformer'):
                base_ckpt[k[len('GPT_Transformer.'):]] = base_ckpt[k]
                del base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                del base_ckpt[k]
            if 'cls_head_finetune' in k:
                del base_ckpt[k]
        return base_ckpt

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            base_ckpt = self._load_checkpoint_dict(base_ckpt)
            
            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(get_missing_parameters_message(incompatible.missing_keys), logger='Transformer')
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(get_unexpected_parameters_message(incompatible.unexpected_keys), logger='Transformer')

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _create_attention_mask(self, L, device, dtype):
        attn_mask = torch.full((L, L), -float("Inf"), device=device, dtype=dtype).to(torch.bool)
        return torch.triu(attn_mask, diagonal=1)

    def forward(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        B, L, _ = group_input_tokens.shape

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        sos_pos = self.sos_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = torch.cat([sos_pos, pos], dim=1)

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / relative_norm
        position = torch.cat([center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        attn_mask = torch.full(
            (L+2, L+2), -float("Inf"), 
            device=group_input_tokens.device, 
            dtype=group_input_tokens.dtype
        ).to(torch.bool)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        ret, encoded_features, distance, idx = self.blocks(center, x, pos, attn_mask, classify=True)

        encoded_features = torch.cat([
            encoded_features[:, 0, :].unsqueeze(1), 
            encoded_features[:, 2:-1, :]
        ], dim=1)

        attn_mask = self._create_attention_mask(L, group_input_tokens.device, group_input_tokens.dtype)

        generated_points = self.generator_blocks(encoded_features, pos_relative, attn_mask, distance, idx)

        neighborhood = neighborhood + center.unsqueeze(2)
        gt_points = neighborhood.reshape(B * self.num_group, self.group_size, 3)

        loss1 = self.loss_func_p1(generated_points, gt_points)
        loss2 = self.loss_func_p2(generated_points, gt_points)

        return ret, loss1 + loss2
