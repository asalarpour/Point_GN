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


def normalize_tensor(data_tensor, knn_tensor, with_center=True):

    if with_center:
        center_tensor = data_tensor.unsqueeze(dim=-2)  # [B, S, 1, C]
        knn_std = torch.std(
            knn_tensor - center_tensor, dim=(0, 1, 3), keepdim=True
        ).clamp(
            min=1e-5
        )  # [1, S, K, D]
        normalized_knn_tensor = (knn_tensor - center_tensor) / knn_std  # [B, S, K, D]
    else:
        knn_std = torch.std(knn_tensor, dim=(0, 1), keepdim=True).clamp(
            min=1e-5
        )  # [1, S, K, D]
        normalized_knn_tensor = knn_tensor / knn_std  # [B, S, K, D]

    return normalized_knn_tensor


# Local Grouper : FPS + k-NN + Normalization
class LocalGrouper(nn.Module):
    def __init__(self, stage_points, k, feat_normalize):
        super().__init__()
        self.stage_points = stage_points
        self.k = k
        self.feat_normalize = feat_normalize

    def forward(self, xyz, feat):
        # xyz:  point coordinates   # [B, N, 3]
        # feat: point features      # [B, N, D]
        b, n, _ = xyz.shape

        # FPS
        if xyz.device == torch.device("cpu"):
            fps_idx = farthest_point_sample(xyz.contiguous(), self.stage_points).long()
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

        # print(torch.abs(feat_sampled.mean(dim=(2), keepdim=True)).mean().item())

        xyz_knn = normalize_tensor(xyz_sampled, xyz_knn, with_center=True)
        # [B, S, K, 3]
        feat_knn = normalize_tensor(
            feat_sampled, feat_knn, with_center=self.feat_normalize
        )
        # [B, S, K, D]

        # Feature Expansion
        b, s, k, _ = feat_knn.shape
        feat_knn = torch.cat(
            [feat_knn, feat_sampled.reshape(b, s, 1, -1).repeat(1, 1, k, 1)], dim=-1
        )  # [B, S, K, D*2]

        return xyz_sampled, feat_sampled, xyz_knn, feat_knn


# GPE Aggregation
class AggregationGPE(nn.Module):
    def __init__(self, out_dim, sigma):
        super().__init__()
        self.gpe_embed = EmbeddingGPE(3, out_dim, sigma)

    def forward(self, xyz_knn, feat_knn):
        # xyz_knn   = [B, S, K, 3]
        # feat_knn  = [B, S, K, D]

        # Geometry Extraction
        position_embed = self.gpe_embed(xyz_knn)  # [B, S, K, out_dim]

        # Weigh
        feat_knn_w = feat_knn + position_embed  # [B, S, K, out_dim]
        feat_knn_w *= position_embed  # [B, S, K, out_dim]

        return feat_knn_w  # [B, S, K, out_dim]


class EmbeddingGPE(nn.Module):
    def __init__(self, in_dim, out_dim, sigma):
        super(EmbeddingGPE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma

        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * self.in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()

        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)

    def forward(self, xyz):
        # xyz = [B, N, 3] or [B, S, K, 3]

        if self.out_dim == 0:
            return xyz

        if xyz.dim() not in {3, 4}:
            raise ValueError("Input must be either [B, in_dim, N] or [B, in_dim, S, K]")

        embeds = []
        # Compute the RBF features for each channel in a loop
        for i in range(self.in_dim):
            tmp = xyz[..., i : i + 1] - self.feat_val.to(xyz.device)
            embed = -0.5 * tmp**2 / (self.sigma**2)
            embeds.append(embed.exp())

        # Concatenate along the last dimension to get all features together
        position_embed = torch.cat(embeds, dim=-1)  # [B, ..., feat_num]

        # Select the required output dimensions using out_idx
        position_embed = torch.index_select(
            position_embed, -1, self.out_idx.to(xyz.device)
        )
        # [B, ..., out_dim]

        # # Reshape based on the original input dimensions
        # if xyz.dim() == 3:
        #     b, _, n = xyz.shape
        #     position_embed = position_embed.permute(0, 2, 1).reshape(b, self.out_dim, n)
        #     # [B, out_dim, N]
        # elif xyz.dim() == 4:
        #     b, _, s, k = xyz.shape
        #     position_embed = position_embed.permute(0, 3, 1, 2).reshape(
        #         b, self.out_dim, s, k
        #     )
        #     # [B, feat_num, S, K]

        return position_embed  # [B, ..., out_dim]


# Neighbor Pooling
class NeighborPooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        # self.out_transform = nn.Sequential(nn.BatchNorm1d(out_dim), nn.GELU())
        self.out_transform = nn.Sequential(nn.GELU())

    def forward(self, feat_knn_w):
        # knn_x_wfeat_knn_w = [B, S, K, D]

        # Feature Aggregation (Pooling)
        feat_agg = feat_knn_w.mean(-2) + feat_knn_w.max(-2)[0]  # [B, S, D]
        feat_agg = self.out_transform(feat_agg.transpose(-2, -1))  # [B, D, S]
        return feat_agg.transpose(-2, -1)  # [B, S, D]


# Non-Parametric Feature Encoder
class EncoderGPECls(nn.Module):
    def __init__(
        self, num_points, init_dim, stages, stage_dim, k, sigma, feat_normalize
    ):
        super().__init__()
        self.num_points = num_points
        self.init_dim = init_dim
        self.stages = stages
        self.stage_dim = stage_dim
        self.sigma = sigma

        # Initial Embedding
        self.init_embed = EmbeddingGPE(3, self.init_dim, sigma)

        self.lg_list = nn.ModuleList()  # FPS, kNN
        self.agpe_list = nn.ModuleList()  # GPE Aggregation
        self.pool_list = nn.ModuleList()  # Pooling

        out_dim = self.init_dim if self.init_dim != 0 else 3
        stage_points = self.num_points

        # Multi-stage Hierarchy
        for i in range(self.stages):
            out_dim = out_dim * 2
            stage_points = stage_points // 2
            self.lg_list.append(LocalGrouper(stage_points, k, feat_normalize))
            self.agpe_list.append(AggregationGPE(out_dim, self.sigma))
            self.pool_list.append(NeighborPooling(out_dim))

    def forward(self, xyz):
        # xyz: point coordinates    # [B, N, 3]

        # Initial Embedding
        feat = self.init_embed(xyz)  # [B, N, init_dim]

        stage_results = []
        skip_feat = feat  # For skip connections

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

            # if i > 0:

            #     # Adjust number of points via interpolation
            #     if skip_feat.shape[1] != feat.shape[1]:
            #         skip_feat = nn.functional.interpolate(
            #             skip_feat.transpose(1, 2), size=feat.shape[1], mode="nearest"
            #         ).transpose(1, 2)

            #     # Since you don't want additional layers, interpolate skip_feat to the exact dimension of feat
            #     if skip_feat.shape[2] != feat.shape[2]:
            #         # Adjust feature dimensions via nearest interpolation
            #         skip_feat = nn.functional.interpolate(
            #             skip_feat, size=feat.shape[2], mode="nearest"
            #         )

            #     # Element-wise addition
            #     feat = feat + skip_feat  # Update feat with skip connection
            # skip_feat = feat  # Update skip_feat for the next stage

            # Stage Pooling
            stage_pooling = torch.cat((feat.max(-2)[0], feat.mean(-2)), dim=1)
            # [B, D_i * 4]

            stage_results.append(stage_pooling)

        # encoded_out = feat.max(-2)[0] + feat.mean(-2)  # [B, dim = 96]
        encoded_out = torch.cat(stage_results, dim=1)  # [B, embed_dim)
        return encoded_out


# Non-Parametric Network
class PointGNCls(nn.Module):
    def __init__(
        self,
        num_points=1024,
        init_dim=6,
        stages=4,
        stage_dim=72,
        k=90,
        sigma=0.3,
        feat_normalize=True,
    ):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncoderGPECls(
            num_points, init_dim, stages, stage_dim, k, sigma, feat_normalize
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
    print(f"Using device: {device}")

    # Create random input data
    random_xyz = torch.randn(batch_size, num_points, in_ch).contiguous().to(device)
    # [B, N, 3]

    # Initialize the model
    model = PointGNCls(num_points=num_points).to(device)

    # Measure the time taken for forward pass
    start_time = time.time()

    # Run the model
    output = model(random_xyz)

    # End time measurement
    end_time = time.time()

    # Print output shape and time taken
    print(f"Output shape: {output.shape}")
    print(f"Time taken for forward pass: {end_time - start_time:.6f} seconds")
