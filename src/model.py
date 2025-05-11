import torch
import torch.nn as nn
from torch_geometric.nn.kge import KGEModel


class Encoder(torch.nn.Module):
    """Concatenate node features with embeddings."""

    def __init__(self, num_nodes: int, embedding_dim: int, features: torch.tensor):
        """
        Args:
            num_nodes (int): Number of nodes in the graph.
            embedding_dim (int): Dimension of the node embeddings.
            features (torch.tensor): Node features of shape (num_nodes, feature_dim).
        """
        super(Encoder, self).__init__()
        assert (
            features.shape[0] == num_nodes
        ), f"Number of nodes and features mismatch. {features.shape[0]} vs {num_nodes}"
        self.features = features
        self.features.requires_grad = False
        if embedding_dim > 0:
            self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        else:
            self.embedding = None

    def to(self, device):
        """Move the encoder to the specified device."""
        super().to(device)
        self.features = self.features.to(device)
        return self

    def forward(self, indices):
        feat = self.features[indices]
        z = [feat]
        if self.embedding is not None:
            z.append(self.embedding(indices))
        return torch.cat(z, dim=1)

    @property
    def embedding_dim(self):
        if self.embedding is None:
            return self.features.shape[1]
        return self.embedding.embedding_dim + self.features.shape[1]

    def reset_parameters(self):
        if self.embedding is not None:
            self.embedding.reset_parameters()

    @property
    def weight(self):
        return self.embedding.weight


def patch_kge_model(model: KGEModel, encoder: nn.Module, encoder_dim: int):
    """
    Patch the KGE model to use the encoder for node embeddings.
    Args:
        model (KGEModel): The KGE model to patch.
        encoder (nn.Module): The encoder to use for node embeddings.
        encoder_dim (int): The dimension of the encoder output.
    """
    model.node_emb = encoder
    model.rel_emb = torch.nn.Embedding(
        model.num_relations, encoder_dim, sparse=model.rel_emb.sparse
    )
    model.hidden_channels = encoder_dim
