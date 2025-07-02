import os
import sys
import pickle
import numpy as np
from torch.utils.data import Dataset

# Path setup
project_path = os.path.abspath(".")
sys.path.append(project_path)

import data.dataset_utils as dutils


def download_modelnet40fewshot(dataset_dir):

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.exists(os.path.join(dataset_dir, "modelnet_fewshot")):
        raise ValueError(
            "MoedlNet40 FewShot cannot be downloaded programmatically through this code. "
            "Please download it manually from its official source and place it in the appropriate directory."
        )


def load_data_modelnet40fewshot(dataset_dir, partition, n_way, k_shot, fold):
    download_modelnet40fewshot(dataset_dir)

    with open(
        os.path.join(
            dataset_dir, "modelnet_fewshot", f"{n_way}way_{k_shot}shot", f"{fold}.pkl"
        ),
        "rb",
    ) as f:
        pkl_data = pickle.load(f)[partition]

    fold_data = np.array([inner[0][:, :3] for inner in pkl_data])
    fold_normals = np.array([inner[0][:, 3:] for inner in pkl_data])
    fold_label = np.array([inner[1] if len(inner) > 1 else None for inner in pkl_data])

    return fold_data, fold_label


class ModelNet40FewShot(Dataset):
    def __init__(
        self,
        dataset_dir,
        num_points,
        partition="train",
        n_way=5,
        k_shot=10,
        augment_type=None,
    ):

        self.num_points = num_points
        self.partition = partition
        self.augment_type = augment_type
        self.fold_num = 0
        all_fold_data = []
        all_fold_label = []
        for fold in range(10):
            fold_data, fold_label = load_data_modelnet40fewshot(
                dataset_dir, partition, n_way, k_shot, fold
            )
            all_fold_data.append(fold_data)
            all_fold_label.append(fold_label)
        self.all_data = np.stack(all_fold_data, axis=0)
        self.all_label = np.stack(all_fold_label, axis=0)

        self.set_fold(self.fold_num)

    def set_augmentation(self, augment_type):
        self.augment_type = augment_type

    def set_fold(self, fold_num):
        self.fold_num = fold_num
        self.data = self.all_data[self.fold_num]
        self.label = self.all_label[self.fold_num]

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
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    modelnet_dir = os.path.join(project_root, "datasets", "modelnet")

    train = ModelNet40FewShot(modelnet_dir, 1024)
    test = ModelNet40FewShot(modelnet_dir, 1024, "test")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        ModelNet40FewShot(modelnet_dir, partition="train", num_points=1024),
        num_workers=4,
        batch_size=32,
        shuffle=False,
        drop_last=True,
    )
    train_loader.dataset.set_fold(2)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(data[0, 0])
        print(
            f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}"
        )

    train_set = ModelNet40FewShot(modelnet_dir, partition="train", num_points=1024)
    test_set = ModelNet40FewShot(modelnet_dir, partition="test", num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
