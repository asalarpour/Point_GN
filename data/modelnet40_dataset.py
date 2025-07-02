import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

# Path setup
project_path = os.path.abspath(".")
sys.path.append(project_path)

import data.dataset_utils as dutils


def download_modelnet40(dataset_dir):

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.exists(os.path.join(dataset_dir, "modelnet40_ply_hdf5_2048")):
        www = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        zipfile = os.path.basename(www)
        os.system("wget %s  --no-check-certificate; unzip %s" % (www, zipfile))
        os.system("mv %s %s" % (zipfile[:-4], dataset_dir))
        os.system("rm %s" % (zipfile))


def load_data_modelnet40(dataset_dir, partition):
    download_modelnet40(dataset_dir)

    all_data = []
    all_label = []
    for h5_name in glob.glob(
        os.path.join(
            dataset_dir, "modelnet40_ply_hdf5_2048", "ply_data_%s*.h5" % partition
        )
    ):

        f = h5py.File(h5_name, "r")
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64")
        f.close()
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, dataset_dir, num_points, partition="train", augment_type=None):
        
        self.num_points = num_points
        self.partition = partition
        self.augment_type = augment_type
        self.data, self.label = load_data_modelnet40(dataset_dir, partition)

    def set_augmentation(self, augment_type):
        self.augment_type = augment_type

    def __getitem__(self, item):
        pointcloud = self.data[item][: self.num_points]
        label = self.label[item]
        if self.partition == "train" and self.augment_type:
            # Check if "translate" is included in augment_type
            if "translate" in self.augment_type:
                pointcloud = dutils.translate_pointcloud(pointcloud)
            # Add more augmentations as needed
            if "scale" in self.augment_type:
                pointcloud = dutils.scale_pointcloud(pointcloud)
            if "rotate" in self.augment_type:
                pointcloud = dutils.rotate_pointcloud(pointcloud)
            if "jitter" in self.augment_type:
                pointcloud = dutils.jitter_pointcloud(pointcloud)
            if "shuffle" in self.augment_type:
                np.random.shuffle(pointcloud)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def normalize_pointcloud(pointcloud):
    # Center the pointcloud by subtracting the mean
    pointcloud = pointcloud - np.mean(pointcloud, axis=0)
    # Scale by the max absolute value of each dimension (to bring it into the [-1, 1] range)
    max_val = np.max(np.abs(pointcloud), axis=0)
    pointcloud = pointcloud / max_val
    return pointcloud


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    modelnet_dir = os.path.join(project_root, "datasets", "modelnet_40")

    train = ModelNet40(modelnet_dir, 1024)
    test = ModelNet40(modelnet_dir, 1024, "test")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        ModelNet40(modelnet_dir, partition="train", num_points=1024),
        num_workers=4,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    for batch_idx, (data, label) in enumerate(train_loader):
        # print(data.max(), data.min())
        print(
            f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}"
        )

    train_set = ModelNet40(modelnet_dir, partition="train", num_points=1024)
    test_set = ModelNet40(modelnet_dir, partition="test", num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
