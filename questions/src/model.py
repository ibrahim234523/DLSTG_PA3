import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Per-sensor temporal convolutions: (B, T, N, 1) -> (B, N, d)."""

    def __init__(self, input_length, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv_1 = nn.Conv1d(1, hidden_dim, kernel_size, padding=padding)
        self.conv_2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.project = nn.Linear(hidden_dim * input_length, hidden_dim)

    def forward(self, history):
        batch_size, input_length, num_nodes, _ = history.shape
        sequences = history.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, 1, input_length)
        features = F.relu(self.conv_1(sequences))
        features = F.relu(self.conv_2(features))                     # (B*N, d, T)
        features = F.relu(self.project(features.flatten(start_dim=1)))  # (B*N, d)
        return features.view(batch_size, num_nodes, -1)


class GraphConv(nn.Module):
    """H = ReLU(A_hat Z W): mixes sensors, independently for each batch item."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.weight = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, node_features, adjacency):
        transformed = self.weight(node_features)                      # (B, N, d)
        propagated = torch.einsum("nm,bmd->bnd", adjacency, transformed)
        return F.relu(propagated)

class FixedAdjacency(nn.Module):
    """Pre-normalised road graph; stored as a buffer, so it never trains."""

    def __init__(self, normalised_adjacency):
        super().__init__()
        self.register_buffer(
            "adjacency", torch.as_tensor(normalised_adjacency, dtype=torch.float32)
        )

    def forward(self):
        return self.adjacency


class AdaptiveAdjacency(nn.Module):
    """A_adaptive = Softmax_row(ReLU(E1 E2^T)), optionally top-k sparsified.

    Top-k is applied BEFORE the softmax: scores outside each row's top-k are set
    to -inf, so the softmax normalises over the kept entries only and every row
    sums to 1. Gradients flow into E1/E2 through the kept scores only; the choice
    of which entries are kept (the indices) is not differentiable.
    """

    def __init__(self, num_nodes, embedding_dim, top_k=None):
        super().__init__()
        self.source_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.target_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.top_k = top_k

    def forward(self):
        scores = F.relu(self.source_embedding @ self.target_embedding.T)    # (N, N)
        if self.top_k is not None:
            kept_values, kept_indices = scores.topk(self.top_k, dim=1)
            scores = torch.full_like(scores, float("-inf")).scatter(
                1, kept_indices, kept_values
            )
        return F.softmax(scores, dim=1)


class HybridAdjacency(nn.Module):
    """A_hybrid = alpha * A_fixed + (1 - alpha) * A_adaptive.

    Both parts are row-stochastic (fixed graph row-normalised, adaptive graph
    softmax-normalised), so the combination is consistent. alpha = sigmoid(theta)
    is learned, initialised at 0.5; set learn_alpha=False to keep it fixed at 0.5.
    """

    def __init__(self, row_normalised_fixed, num_nodes, embedding_dim, top_k=None,
                 learn_alpha=True):
        super().__init__()
        self.fixed = FixedAdjacency(row_normalised_fixed)
        self.adaptive = AdaptiveAdjacency(num_nodes, embedding_dim, top_k)
        initial_theta = torch.zeros(1)                                # sigmoid(0) = 0.5
        if learn_alpha:
            self.alpha_logit = nn.Parameter(initial_theta)
        else:
            self.register_buffer("alpha_logit", initial_theta)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self):
        return self.alpha * self.fixed() + (1.0 - self.alpha) * self.adaptive()


# --------------------------------------------------------------------------
# Full forecaster
# --------------------------------------------------------------------------
class TrafficForecaster(nn.Module):
    """Temporal encoder + optional graph convolution + forecast head."""

    def __init__(self, adjacency_provider, input_length, horizon, hidden_dim):
        super().__init__()
        self.encoder = TemporalEncoder(input_length, hidden_dim)
        self.adjacency_provider = adjacency_provider          # None => temporal-only
        self.graph_conv = GraphConv(hidden_dim) if adjacency_provider is not None else None
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, horizon)
        )

    def forward(self, history):
        node_features = self.encoder(history)                          # (B, N, d)
        if self.graph_conv is not None:
            node_features = self.graph_conv(node_features, self.adjacency_provider())
        forecast = self.head(node_features)                            # (B, N, horizon)
        return forecast.permute(0, 2, 1)                               # (B, horizon, N)


def build_model(variant, fixed_adjacency, input_length, horizon, hidden_dim,
                embedding_dim, top_k, learn_alpha=True):
    """variant in {'temporal', 'fixed', 'adaptive', 'hybrid'}.

    fixed_adjacency: raw (N, N) numpy road adjacency (before normalisation).
    """
    # imported here to keep model.py importable without the data stack
    from data import normalise_adjacency

    num_nodes = fixed_adjacency.shape[0]
    is_symmetric = bool(np.allclose(fixed_adjacency, fixed_adjacency.T))

    if variant == "temporal":
        provider = None
    elif variant == "fixed":
        mode = "sym" if is_symmetric else "row"        # directed graphs: row-normalise
        provider = FixedAdjacency(normalise_adjacency(fixed_adjacency, mode))
    elif variant == "adaptive":
        provider = AdaptiveAdjacency(num_nodes, embedding_dim, top_k)
    elif variant == "hybrid":
        provider = HybridAdjacency(
            normalise_adjacency(fixed_adjacency, "row"),
            num_nodes, embedding_dim, top_k, learn_alpha,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return TrafficForecaster(provider, input_length, horizon, hidden_dim)
