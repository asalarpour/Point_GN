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


def download_scanobjectnn(dataset_dir):

    if not os.path.exists(os.path.join(dataset_dir, "h5_files")):

        raise ValueError(
            "ScanObjectNN cannot be downloaded programmatically through this code. "
            "Please download it manually from its official source and place it in the appropriate directory."
        )


def load_data_scanobjectnn(dataset_dir, partition, split):
    download_scanobjectnn(dataset_dir)

    if partition == "train":
        partition = "training"

    if split == "OBJ_BG":
        h5_name = os.path.join(
            dataset_dir, "h5_files", "main_split", f"{partition}_objectdataset.h5"
        )
    elif split == "OBJ_ONLY":
        h5_name = os.path.join(
            dataset_dir, "h5_files", "main_split_nobg", f"{partition}_objectdataset.h5"
        )

    elif split == "PB_T50_RS":
        h5_name = os.path.join(
            dataset_dir,
            "h5_files",
            "main_split",
            f"{partition}_objectdataset_augmentedrot_scale75.h5",
        )

    all_data = []
    all_label = []
    f = h5py.File(h5_name, mode="r")
    data = f["data"][:].astype("float32")
    label = f["label"][:].astype("int64")
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


class ScanObjectNN(Dataset):
    def __init__(
        self,
        dataset_dir,
        num_points,
        partition="train",
        split="PB_T50_RS",
        augment_type=None,
    ):
        
        self.num_points = num_points
        self.partition = partition
        self.augment_type = augment_type
        self.data, self.label = load_data_scanobjectnn(dataset_dir, partition, split)

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

        np.random.shuffle(pointcloud)
        # pointcloud = normalize_pointcloud(pointcloud)

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
    modelnet_dir = os.path.join(project_root, "datasets", "scanobjectnn")

    # train = ScanObjectNN(modelnet_dir, 1024)
    test = ScanObjectNN(modelnet_dir, 1024, "test")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        ScanObjectNN(modelnet_dir, partition="train", num_points=1024),
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

    train_set = ScanObjectNN(modelnet_dir, partition="train", num_points=1024)
    test_set = ScanObjectNN(modelnet_dir, partition="test", num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
