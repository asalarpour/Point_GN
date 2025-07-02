import os
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import general_utils as gutils
from models.point_gn import PointGNCls
from models.point_nn import PointNNCls
from models.point_nn_original import Point_NN_Original


def process_data(data_loader, model, device):
    features_list, labels_list = [], []

    # Loop through the provided data loader
    for points, labels in tqdm(data_loader, leave=False):
        # points: [B, num_points, 3]
        point_features = model(points.to(device))  # [B, num_features]
        features_list.append(point_features)

        labels = labels.to(device)
        labels_list.append(labels)

    features = torch.cat(features_list, dim=0)  # [num_samples, num_features]
    features = F.normalize(features, dim=-1)
    # features = features.permute(1, 0)  # [num_features, num_samples]

    labels = torch.cat(labels_list, dim=0)  # [num_samples, 1]

    return features, labels


def get_model_and_param(args):
    if args.model == "pointgn":
        args.seed = 42
        args.init_dim = 27
        args.stage_dim = 27
        args.stages = 4
        args.k = 120

        if args.dataset == "modelnet40":
            args.sigma = 0.4

        elif args.dataset == "scanobject":
            args.sigma = 0.3

        model = PointGNCls(
            num_points=args.num_points,
            init_dim=args.init_dim,
            stages=args.stages,
            stage_dim=args.stage_dim,
            k=args.k,
            sigma=args.sigma,
            feat_normalize=args.feat_normalize,
        )

    elif args.model == "pointnn":
        args.seed = 42
        args.alpha = 1000
        args.beta = 100
        args.init_dim = 72
        args.stage_dim = 72
        args.stages = 4
        args.k = 90

        model = PointNNCls(
            num_points=args.num_points,
            init_dim=args.init_dim,
            stages=args.stages,
            stage_dim=args.stage_dim,
            k=args.k,
            alpha=args.alpha,
            beta=args.beta,
        )
        model = Point_NN_Original(
            input_points=args.num_points,
            num_stages=args.stages,
            embed_dim=args.stage_dim,
            k_neighbors=args.k,
            alpha=args.alpha,
            beta=args.beta,
        )

    return model


def process_and_evaluate(train_loader, test_loader, model, device):
    # Process training data
    start_train_time = time.time()
    train_features, train_labels = process_data(train_loader, model, device)
    train_labels = F.one_hot(train_labels).squeeze().float()
    train_time = time.time() - start_train_time

    # Process testing data
    start_test_time = time.time()
    test_features, test_labels = process_data(test_loader, model, device)
    test_time = time.time() - start_test_time

    # Calculate accuracies
    acc_cos, gamma = gutils.cosine_similarity(
        test_features, train_features, train_labels, test_labels
    )
    acc_1nn = gutils.one_nn_classification(
        test_features, train_features, train_labels, test_labels
    )

    # Return results
    return {
        "train_time": train_time,
        "test_time": test_time,
        "acc_1nn": acc_1nn,
        "acc_cos": acc_cos,
        "gamma": gamma,
    }


@torch.no_grad()
def main_cls():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    dataset_dir = os.path.join(project_root, "datasets")

    args = gutils.get_arguments()
    train_loader, test_loader = gutils.get_cls_dataloader(dataset_dir, args)

    gutils.set_seed(args.seed)

    model = get_model_and_param(args)

    model.to(device).eval()
    start_total_time = time.time()

    if args.dataset == "modelnet40fewshot":
        train_times = []
        test_times = []
        acc_cos = []
        acc_1nn = []
        for fold in tqdm(range(10)):
            train_loader.dataset.set_fold(fold)
            test_loader.dataset.set_fold(fold)

            results = process_and_evaluate(train_loader, test_loader, model, device)
            train_times.append(results["train_time"])
            test_times.append(results["test_time"])
            acc_cos.append(results["acc_cos"])
            acc_1nn.append(results["acc_1nn"])
        results = {
            "train_time": np.mean(train_times),
            "test_time": np.mean(test_times),
            "acc_cos": np.mean(acc_cos),
            "acc_1nn": np.mean(acc_1nn),
            "gamma": 0,
        }
    else:

        results = process_and_evaluate(train_loader, test_loader, model, device)

    total_time = time.time() - start_total_time

    print("==============================")
    print(f"model = {args.model}")
    print(f"dataset = {args.dataset}")
    print(f"seed = {args.seed}")
    print(f"k = {args.k}")
    print(f"init_dim = {args.init_dim}")
    print(f"stages = {args.stages}")
    print(f"stage_dim = {args.stage_dim}")
    print(f"total_time = {total_time}")
    print(f"train_time = {results['train_time']}")
    print(f"test_time = {results['test_time']}")
    print(f"acc_cos = {results['acc_cos']}")
    print(f"acc_1nn = {results['acc_1nn']}")
    print(f"gamma = {results['gamma']}")


if __name__ == "__main__":
    main_cls()
