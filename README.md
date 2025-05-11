# Knowledge Graph Embeddings with Node Features

A user-friendly toolkit for training Knowledge Graph Embedding (KGE) models enhanced with additional node features. This project builds on top of [PyTorch Geometric's KGE modules](https://pytorch-geometric.readthedocs.io) and provides an easy way to combine learned node embeddings with your own node-level feature vectors.

---

## 📦 Features

* **Plug-and-play Encoder**: A simple `Encoder` module to concatenate node embeddings with your custom features.
* **Model Patching**: Utilities to seamlessly patch existing KGE models (TransE, DistMult, RotatE) to use the new encoder.
* **Flexible Training Script**: A command-line `train.py` that supports different KGE models, optional linear transformations, and custom feature files.

---

## 🔧 Installation

1. **Prerequisites**

   * Python 3.8+
   * [PyTorch](https://pytorch.org/) with CUDA support (optional but recommended)
   * [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
   * [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. **Install dependencies**

   ```bash
   uv install
   ```

---

## 🚀 Quick Start

### 1. Prepare Your Node Features

* Create or load a tensor of node features of shape `(num_nodes, feature_dim)`.
* Save it as a PyTorch file:

  ```python
  import torch
  features = torch.randn(num_nodes, feature_dim)
  torch.save({'features': features}, 'data/features.pt')
  ```

### 2. Initialize and Patch a KGE Model

```python
from torch_geometric.nn.kge import DistMult
from src.model import Encoder, patch_kge_model
import torch

# 1. Create baseline KGE model (e.g., DistMult)
model = DistMult(num_nodes=100, num_relations=10, hidden_channels=64)

# 2. Load your features
features = torch.load('data/features.pt')['features']

# 3. Create the Encoder and patch the model
encoder = Encoder(num_nodes=100, embedding_dim=64, features=features)
patch_kge_model(model, encoder, encoder.embedding_dim)

# 4. Test a forward pass
head = torch.tensor([0, 1, 2])
rel = torch.tensor([0, 1, 0])
tail = torch.tensor([1, 2, 3])
print(model(head, rel, tail))  # Now uses combined embeddings + features
```

---

## 🎓 Training Your Model

Use the provided `train.py` script to train on the standard FB15k-237 dataset or your own data.

```bash
uv run src/train.py \
  --model distmult \
  --features data/features.pt \
  --embedding_dim 128 \
  --use_linear
```

### Command-Line Options

| Option            | Description                                                     | Example                       |
| ----------------- | --------------------------------------------------------------- | ----------------------------- |
| `--model`         | KGE model to use (`transe`, `distmult`, `rotate`)               | `--model distmult`            |
| `--features`      | Path to node features file (`.pt` format)                       | `--features data/features.pt` |
| `--embedding_dim` | Base embedding dimension (before concatenating features)        | `--embedding_dim 128`         |
| `--use_linear`    | Apply an additional linear layer after concatenation (optional) | `--use_linear`                |

### What Happens During Training?

1. **Loading Data**: Automatically downloads FB15k-237 and splits into train/val/test.
2. **Model Setup**: Builds your chosen KGE model and patches node embeddings with your `Encoder`.
3. **Optimization**: Uses standard optimizers (Adam) with model-specific learning rates.
4. **Evaluation**: Reports Mean Rank, Mean Reciprocal Rank (MRR), and Hits\@10 on the test set.
