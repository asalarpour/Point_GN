import numpy as np


def random_point_dropout(pc, max_dropout_ratio=0.875):
    """batch_pc: BxNx3"""
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud
