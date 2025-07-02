
# 🚀 Point‑GN: A Non‑Parametric Network Using Gaussian Positional Encoding for Point Cloud Classification

[![Papers with Code Badge – Training‑Free 3D](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-gn-a-non-parametric-network-using/training-free-3d-point-cloud-classification)](https://paperswithcode.com/sota/training-free-3d-point-cloud-classification?p=point-gn-a-non-parametric-network-using)
[![Papers with Code Badge – Few‑Shot](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-gn-a-non-parametric-network-using/training-free-3d-point-cloud-classification-1)](https://paperswithcode.com/sota/training-free-3d-point-cloud-classification-1?p=point-gn-a-non-parametric-network-using)

**Official implementation** of the WACV 2025 paper:  
**“Point‑GN: A Non‑Parametric Network Using Gaussian Positional Encoding for Point Cloud Classification”** by Marzieh Mohammadi & Amir Salarpour.  
📄 [View Paper](https://openaccess.thecvf.com/content/WACV2025/html/Mohammadi_Point-GN_A_Non-Parametric_Network_using_Gaussian_Positional_Encoding_for_Point_WACV_2025_paper.html)

---

## 🧠 Overview

**Point‑GN** is a training-free, non-parametric model for 3D point cloud classification.  
It combines:

- 🧮 Farthest Point Sampling (FPS)
- 🔗 k-Nearest Neighbors (k-NN)
- 🌐 Gaussian Positional Encoding (GPE)

It achieves competitive accuracy without a single trainable parameter.

---

## 📁 Repository Structure

```
Point-GN/
├── general_utils.py         # Utility functions
├── train_free_main.py       # Main entry point
├── models/                  # GPE & Point-NN model code
├── data/                    # Dataset loaders
├── pointnet2_ops/           # Compiled PointNet++ ops
└── README.md
```

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/asalarpour/Point-GN.git
cd Point-GN

# Install core dependencies
pip install torch torchvision
pip install -r requirements.txt

# Compile PointNet++ ops
cd pointnet2_ops
pip install .
cd ..
```

---

## 📦 Supported Datasets

* ModelNet40
* ScanObjectNN
* ModelNet40FewShot (for few-shot tasks)

**Directory layout:**

```
datasets/
├── modelnet40/
│   ├── modelnet40_ply_hdf5_2048/
│   └── modelnet_fewshot/
│       └── 5way_10shot/
└── scanobjectnn/
    └── objectdataset_*.h5
```

---

## 🎯 Usage

**Standard classification (ModelNet40):**

```bash
python train_free_main.py --model pointgn --dataset modelnet40
```

**ScanObjectNN (OBJ_BG split):**

```bash
python train_free_main.py --model pointgn --dataset scanobject --split OBJ_BG
```

**Few‑shot classification (5‑way 10‑shot):**

```bash
python train_free_main.py --model pointgn --dataset modelnet40fewshot --n_way 5 --k_shots 10
```

### 🔧 Key Arguments

| Argument       | Description                                         |
| -------------- | --------------------------------------------------- |
| `--model`      | `pointgn` or `pointnn`                              |
| `--dataset`    | `modelnet40`, `scanobject`, `modelnet40fewshot`     |
| `--split`      | For ScanObjectNN: `OBJ_BG`, `OBJ_ONLY`, `PB_T50_RS` |
| `--n_way`      | Number of classes in few-shot tasks                 |
| `--k_shots`    | Number of examples per class                        |
| `--num-points` | Number of points per shape                          |
| `--sigma`      | Gaussian kernel scale                               |

---

## 🧩 How It Works

### Gaussian Positional Encoding (GPE)

Each 3D coordinate is mapped using Gaussian kernels centered at fixed reference points:

```
γ_x(x_i, v_j) = exp(−||x_i − v_j||² / (2σ²))
```

(similarly for `y` and `z` axes).

### Pipeline Summary

1. **FPS + k-NN** grouping
2. Apply **GPE** to local groups
3. Hierarchical **pooling**
4. **Similarity-based** classification (e.g., 1-NN, cosine)

---

## 📊 Results

| Dataset      | Accuracy | Parameters |
| ------------ | -------: | ---------- |
| ModelNet40   |    85.3% | 0          |
| ScanObjectNN |    85.9% | 0          |

> No training needed — fast, efficient, and competitive with state-of-the-art parametric models.

---

## 🧪 Tuning Tips

* Recommended `σ`:

  * 0.4 for ModelNet40
  * 0.3 for ScanObjectNN
* Larger σ smooths features; smaller σ captures finer details.

---

## 📝 Citation

```bibtex
@InProceedings{Mohammadi_2025_WACV,
  author    = {Mohammadi, Marzieh and Salarpour, Amir},
  title     = {Point-GN: A Non-Parametric Network Using Gaussian Positional Encoding for Point Cloud Classification},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2025},
  pages     = {3487--3496}
}
```

---

## 📬 Contact

📧 Questions? Reach out to: **[asalarp@clemson.com](mailto:asalarp@clemson.com)**

---

## 🙌 Acknowledgements

Inspired by:

* [Point-NN (Zhang et al., 2023)](https://arxiv.org/abs/2303.08134)
* [PointNet++](https://arxiv.org/abs/1706.02413)