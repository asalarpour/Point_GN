import os
import sys
import math
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

# Path setup
project_path = os.path.abspath(".")
sys.path.append(project_path)

from models.model_utils import index_points, knn_point, farthest_point_sample


# Local Grouper : FPS + k-NN + Normalization
class LocalGrouper(nn.Module):
    def __init__(self, stage_points, k):
        super().__init__()
        self.stage_points = stage_points
        self.k = k

    def forward(self, xyz, feat):
        # xyz:  point coordinates   # [B, N, 3]
        # feat: point features      # [B, N, D]
        b, n, _ = xyz.shape

        # FPS
        if xyz.device == torch.device('cpu'):
            fps_idx = farthest_point_sample(
                xyz.contiguous(), self.stage_points
            ).long()
        else:
            fps_idx = pointnet2_utils.furthest_point_sample(
                xyz.contiguous(), self.stage_points
            ).long()
        xyz_sampled = index_points(xyz, fps_idx)  # [B, S, 3]
        feat_sampled = index_points(feat, fps_idx)  # [B, S, D]

        # kNN
        idx_knn = knn_point(self.k, xyz, xyz_sampled)
        xyz_knn = index_points(xyz, idx_knn)  # [B, S, K, 3]
        feat_knn = index_points(feat, idx_knn)  # [B, S, K, D]

        # Normalize feat (features) and xyz (coordinates)
        xyz_center = xyz_sampled.unsqueeze(dim=-2)  # [B, S, 1, 3]
        xyz_std = torch.std(xyz_knn - xyz_center)  # [1]
        xyz_knn = (xyz_knn - xyz_center) / (xyz_std + 1e-5)  # [B, S, K, 3]

        feat_center = feat_sampled.unsqueeze(dim=-2)  # [B, S, 1, D]
        feat_std = torch.std(feat_knn - feat_center)  # [1]
        feat_knn = (feat_knn - feat_center) / (feat_std + 1e-5)  # [B, S, K, D]

        # Feature Expansion
        b, s, k, _ = feat_knn.shape
        feat_knn = torch.cat(
            [feat_knn, feat_sampled.reshape(b, s, 1, -1).repeat(1, 1, k, 1)], dim=-1
        )  # [B, S, K, D*2]

        return xyz_sampled, feat_sampled, xyz_knn, feat_knn


# GPE Aggregation
class AggregationSPE(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.gpe_embed = EmbeddingSPE(3, out_dim, alpha, beta)

    def forward(self, xyz_knn, feat_knn):
        # xyz_knn   = [B, S, K, 3]
        # feat_knn  = [B, S, K, D]

        # Geometry Extraction
        position_embed = self.gpe_embed(xyz_knn)  # [B, S, K, out_dim]

        # Weigh
        feat_knn_w = feat_knn + position_embed  # [B, S, K, out_dim]
        feat_knn_w *= position_embed  # [B, S, K, out_dim]

        return feat_knn_w  # [B, S, K, out_dim]


class EmbeddingSPE(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super(EmbeddingSPE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.beta = beta

        feat_dim = math.ceil(out_dim / (in_dim * 2))
        self.feat_num = feat_dim * 2 * self.in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()

        self.feat_val = torch.arange(feat_dim).float()
        self.dim_embed = torch.pow(alpha, self.feat_val / feat_dim)

    def forward(self, xyz):
        # xyz = [B, N, 3] or [B, S, K, 3]

        if self.out_dim == 0:
            return xyz

        if xyz.dim() not in {3, 4}:
            raise ValueError("Input must be either [B, in_dim, N] or [B, in_dim, S, K]")
      
        
        embeds = []
        # Compute the RBF features for each channel in a loop
        for i in range(self.in_dim):
            tmp = self.beta * xyz[..., i : i + 1]
            div_embed = torch.div(tmp, self.dim_embed.to(xyz.device))
            sin_embed = torch.sin(div_embed)  # [B, N, feat_dim]
            cos_embed = torch.cos(div_embed)  # [B, N, feat_dim]
            embeds.append(torch.stack([sin_embed, cos_embed], dim=-1).flatten(-2))

        # Concatenate along the last dimension to get all features together
        position_embed = torch.cat(embeds, dim=-1)  # [B, ..., feat_num]

        # Select the required output dimensions using out_idx
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))
        # [B, ..., out_dim]

        return position_embed  # [B, ..., out_dim]


# Neighbor Pooling
class NeighborPooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(nn.BatchNorm1d(out_dim), nn.GELU())

    def forward(self, feat_knn_w):
        # knn_x_wfeat_knn_w = [B, S, K, D]

        # Feature Aggregation (Pooling)
        feat_agg = feat_knn_w.max(-2)[0] + feat_knn_w.mean(-2)  # [B, S, D]
        feat_agg = self.out_transform(feat_agg.transpose(-2, -1))  # [B, D, S]
        return feat_agg.transpose(-2, -1)  # [B, S, D]


# Non-Parametric Feature Encoder
class EncoderSPECls(nn.Module):
    def __init__(self, num_points, init_dim, stages, stage_dim, k, alpha, beta):
        super().__init__()
        self.num_points = num_points
        self.init_dim = init_dim
        self.stages = stages
        self.stage_dim = stage_dim
        self.alpha = alpha
        self.beta = beta

        # Initial Embedding
        self.init_embed = EmbeddingSPE(3, self.init_dim, alpha, beta)

        self.lg_list = nn.ModuleList()  # FPS, kNN
        self.agpe_list = nn.ModuleList()  # GPE Aggregation
        self.pool_list = nn.ModuleList()  # Pooling

        out_dim = self.init_dim if self.init_dim != 0 else 3
        stage_points = self.num_points

        # Multi-stage Hierarchy
        for i in range(self.stages):
            out_dim = out_dim * 2
            stage_points = stage_points // 2
            self.lg_list.append(LocalGrouper(stage_points, k))
            self.agpe_list.append(AggregationSPE(out_dim, alpha, beta))
            self.pool_list.append(NeighborPooling(out_dim))

    def forward(self, xyz):
        # xyz: point coordinates    # [B, N, 3]

        # Initial Embedding
        feat = self.init_embed(xyz)  # [B, N, init_dim]

        stage_results = []

        # Multi-stage Hierarchy
        for i in range(self.stages):
            # FPS, kNN
            xyz, feat, xyz_knn, feat_knn = self.lg_list[i](xyz, feat)
            # xyz:      [B, N/2^i, 3]
            # feat:     [B, N/2^i, D_i]
            # xyz_knn:  [B, N/2^i, K, 3]
            # feat_knn: [B, N/2^i, K, D_i]

            # GPE Aggregation
            feat_knn_w = self.agpe_list[i](xyz_knn, feat_knn)
            # [B, N/2^i, K, D_i * 2]

            # Neighbor Pooling
            feat = self.pool_list[i](feat_knn_w)  # [B, N/2^i, D_i * 2]

        # after finish 4 stage loop x  = [B, 64, 96]

        # Global Pooling
        encoded_out = feat.max(-2)[0] + feat.mean(-2)  # [B, dim = 96]
        return encoded_out


# Non-Parametric Network
class PointNNCls(nn.Module):
    def __init__(
        self,
        num_points=1024,
        init_dim=6,
        stages=4,
        stage_dim=72,
        k=90,
        alpha=1000,
        beta=100,
    ):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncoderSPECls(
            num_points, init_dim, stages, stage_dim, k, alpha, beta
        )

    def forward(self, xyz):
        # xyz: point coordinates # [B, N, 3]

        # Non-Parametric Encoder
        x = self.EncNP(xyz)  # [B, embed_dim]
        return x


import time

if __name__ == "__main__":
    # Parameters
    batch_size = 128
    num_points = 1024
    in_ch = 3

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Using device: {device}")

    # Create random input data
    random_xyz = torch.randn(batch_size, num_points, in_ch).contiguous().to(device)
    # [B, N, 3]

    # Initialize the model
    model = PointNNCls(num_points=num_points).to(device)

    # Measure the time taken for forward pass
    start_time = time.time()

    # Run the model
    output = model(random_xyz)

    # End time measurement
    end_time = time.time()

    # Print output shape and time taken
    print(f"Output shape: {output.shape}")
    print(f"Time taken for forward pass: {end_time - start_time:.6f} seconds")
