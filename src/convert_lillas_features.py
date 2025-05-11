# coding: utf-8
import json

import numpy as np
import torch
from torch_geometric.datasets import FB15k_237

PROTOTYPES_PATH = "data/prototype_iterations_5_manh.json"

# Translate node ids to idxes according to the torch-geometric dataset
# using https://gist.github.com/ricsi98/81138cd51e8fe7e15644805c2371bca0
ds = FB15k_237(root="data/FB15k")

node_dict: dict[str, int] = {}
rel_dict: dict[str, int] = {}

for path in ds.raw_paths:
    with open(path, "r") as f:
        lines = [x.split("\t") for x in f.read().split("\n")[:-1]]

    for i, (src, rel, dst) in enumerate(lines):
        if src not in node_dict:
            node_dict[src] = len(node_dict)
        if dst not in node_dict:
            node_dict[dst] = len(node_dict)
        if rel not in rel_dict:
            rel_dict[rel] = len(rel_dict)


# Parse prototype data
with open(PROTOTYPES_PATH, "r") as f:
    data = json.load(f)

# Load the final iteration of prototypes
prototype_mapping = {v: k for k, items in data[-1]["assignments"].items() for v in items}

# Create one-hot encoding for prototypes
ordered_node_ids = sorted(node_dict.items(), key=lambda x: x[1])
missing_mappings = set(node_dict.keys()) - set(prototype_mapping.keys())

if len(missing_mappings) > 0:
    print(
        f"Warning: {len(missing_mappings)} nodes are missing in the prototype mapping. Filling with zeros."
    )
    for node in missing_mappings:
        prototype_mapping[node] = "NoPrototype"

ordered_prototypes = sorted(prototype_mapping.items(), key=lambda x: node_dict[x[0]])

prototype_idxes = {p: idx for idx, p in enumerate(set(prototype_mapping.values()))}
ohe_codes_table = np.eye(len(prototype_idxes))

ohe_codes = np.stack(
    [ohe_codes_table[prototype_idxes[p]] for _, p in ordered_prototypes],
    axis=0,
)

# Save the one-hot encoding with torch
ohe_torch = torch.tensor(ohe_codes, dtype=torch.float32)
save_path = PROTOTYPES_PATH.replace(".json", "-features.pt")
print("Saving one-hot encoding to", save_path)
torch.save(
    {"features": ohe_torch, "prototype_index": prototype_idxes},
    save_path,
)
