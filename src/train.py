import argparse
import tempfile

import torch
import torch.optim as optim
from torch.nn import Linear, Sequential
from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import DistMult, RotatE, TransE

from model import Encoder, patch_kge_model

model_map = {
    "transe": TransE,
    "distmult": DistMult,
    "rotate": RotatE,
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=model_map.keys(), type=str.lower, required=True)
# Path where the node features are located
parser.add_argument("--features", type=str, required=False)
# Binary flag whether to use Linear in the encoder or not
parser.add_argument("--use_linear", action="store_true", help="Use Linear in the encoder")
parser.add_argument("--embedding_dim", type=int, default=50, help="Embedding dimension")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
path = "data/FB15k"

train_data = FB15k_237(path, split="train")[0].to(device)
val_data = FB15k_237(path, split="val")[0].to(device)
test_data = FB15k_237(path, split="test")[0].to(device)

model_arg_map = {"rotate": {"margin": 9.0}}
model = model_map[args.model](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=args.embedding_dim,
    **model_arg_map.get(args.model, {}),
)

# Prepare the encoder to use features
if args.features:
    features = torch.load(args.features, weights_only=True)["features"]
    enc = Encoder(
        num_nodes=train_data.num_nodes,
        embedding_dim=args.embedding_dim,
        features=features,
    ).to(device)

    steps = [enc]

    if args.use_linear:
        steps.append(Linear(enc.embedding_dim, enc.embedding_dim))

    seq = Sequential(*steps).to(device)

    patch_kge_model(model, seq, enc.embedding_dim)
    print("Patched model with new encoder:")
    print("\t- features shape:", features.shape)
    print("\t- encoder embedding dimensions:", enc.embedding_dim)
else:
    print("No features provided, using default encoder")

model = model.to(device)

loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=1000,
    shuffle=True,
)

optimizer_map = {
    "transe": optim.Adam(model.parameters(), lr=0.01),
    "distmult": optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6),
    "rotate": optim.Adam(model.parameters(), lr=1e-3),
}
optimizer = optimizer_map[args.model]


def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=20000,
        k=10,
    )


best_hits = 0
with tempfile.TemporaryDirectory() as tmpdir:
    for epoch in range(1, args.epochs + 1):
        loss = train()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
        if epoch % 25 == 0:
            rank, mrr, hits = test(val_data)
            print(
                f"Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, "
                f"Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}"
            )
            if hits > best_hits:
                best_hits = hits
                torch.save(
                    {"model": model.state_dict()},
                    f"{tmpdir}/best_model.pt",
                )
                print(f"Saved new best model with Hits@10: {hits:.4f}")

    # load the best model
    print("Loading best model with Hits@10:", best_hits)
    model.load_state_dict(torch.load(f"{tmpdir}/best_model.pt")["model"])


rank, mrr, hits_at_10 = test(test_data)
print(f"Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, " f"Test Hits@10: {hits_at_10:.4f}")
