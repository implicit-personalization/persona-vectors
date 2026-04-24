import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


def pairwise_cosine_similarity(
    vectors: list[torch.Tensor], center: bool = False
) -> torch.Tensor:
    """Compute pairwise cosine similarity between vectors.

    Args:
        vectors: List of 1-D tensors (one per persona/condition).
        center: If True, subtract the mean vector across the list before
            normalising. LLM residual-stream means share a large DC component
            that pushes every pairwise cosine toward ~1; centering removes it
            so the remaining structure (which personas actually cluster) shows.
    """

    if not vectors:
        raise ValueError("vectors must not be empty")
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("vectors must be 1-D tensors")

    stacked = torch.stack([vector.float() for vector in vectors])
    if center:
        stacked = stacked - stacked.mean(dim=0, keepdim=True)
    normalized = F.normalize(stacked, dim=1)
    return normalized @ normalized.T


def project_pca(samples: torch.Tensor) -> torch.Tensor:
    """Project samples to 2D using PCA.

    Args:
        samples: Tensor with shape (n_samples, hidden_size).

    Returns:
        Tensor with shape (n_samples, 2).
    """
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")

    embedding = PCA(n_components=2).fit_transform(samples.float().cpu().numpy())
    return torch.from_numpy(embedding)


def project_umap(samples: torch.Tensor) -> torch.Tensor:
    """Project samples to 2D using UMAP.

    Args:
        samples: Tensor with shape (n_samples, hidden_size).

    Returns:
        Tensor with shape (n_samples, 2).
    """
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")

    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn is required for UMAP projections") from exc

    embedding = umap.UMAP(n_components=2, random_state=1337).fit_transform(
        samples.float().cpu().numpy()
    )
    return torch.from_numpy(embedding)


def build_embedding_figure(
    coords: torch.Tensor,
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    hover_text: list[str] | None = None,
) -> go.Figure:
    """Build a 2D scatter plot from projected coordinates."""
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_samples, 2)")
    if len(labels) != coords.shape[0]:
        raise ValueError("labels must match number of samples")
    if hover_text is not None and len(hover_text) != coords.shape[0]:
        raise ValueError("hover_text must match number of samples")

    fig = go.Figure()
    unique_labels = list(dict.fromkeys(labels))

    for label in unique_labels:
        mask = torch.tensor([value == label for value in labels], dtype=torch.bool)
        selected = coords[mask]
        fig.add_trace(
            go.Scatter(
                x=selected[:, 0].tolist(),
                y=selected[:, 1].tolist(),
                mode="markers",
                name=label,
                showlegend=False,
                marker=dict(
                    size=8,
                    opacity=0.8,
                ),
                text=(
                    [hover_text[i] for i, value in enumerate(labels) if value == label]
                    if hover_text is not None
                    else None
                ),
                hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    return fig


def pca_explained_variance(
    samples: torch.Tensor, n_components: int | None = None
) -> np.ndarray:
    """Return the explained variance ratio for each principal component."""

    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, hidden_size)")

    X = samples.float().cpu().numpy()
    max_components = min(X.shape)
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components).fit(X)
    return pca.explained_variance_ratio_
