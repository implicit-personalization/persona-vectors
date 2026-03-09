import json

import plotly.graph_objects as go
import torch
import torch.nn.functional as F

from src.environment import get_artifacts_dir


def _plots_dir():
    path = get_artifacts_dir() / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_layer_similarity(
    short: torch.Tensor,
    long: torch.Tensor,
    title: str = "Layer-wise Activation Similarity",
    filename: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot cosine similarity between two sets of activation across layers.

    Args:
        short: (L, d_model) tensor of activations for the first prompt.
        long: (L, d_model) tensor of activations for the second prompt.
        title: Plot title.
        filename: If provided, save an interactive HTML file as
            <artifacts_dir>/plots/<filename>.html.
        show: If True, open the plot in the browser.

    Returns:
        The Plotly figure object.
    """
    similarities = F.cosine_similarity(short, long, dim=1).tolist()
    layers = list(range(len(similarities)))

    fig = go.Figure(
        go.Scatter(
            x=layers,
            y=similarities,
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(color="blue"),
            hovertemplate="Layer %{x}<br>Cosine sim: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Cosine similarity",
        hovermode="x",
        template="plotly_white",
    )

    if filename is not None:
        output_path = _plots_dir() / f"{filename}.html"
        fig.write_html(str(output_path))
        print(f"Plot saved to {output_path}")

    if show:
        fig.show()

    return fig


def layerwise_cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity per layer for two (n_layers, d_model) tensors."""
    if vec_a.ndim != 2 or vec_b.ndim != 2:
        raise ValueError("inputs must have shape (n_layers, d_model)")
    if vec_a.shape != vec_b.shape:
        raise ValueError("inputs must share shape")
    return F.cosine_similarity(vec_a, vec_b, dim=1)


def pca_project_personas(
    persona_vectors: dict[str, torch.Tensor],
    layer: int,
    n_components: int = 2,
    center: bool = True,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Project persona vectors at a given layer with PCA.

    Args:
        persona_vectors: Mapping persona_id -> (n_layers, d_model) tensor.
        layer: Layer index to project.
        n_components: Number of principal components.
        center: If True, center vectors before PCA.

    Returns:
        names: Persona ids in matrix order.
        coords: (n_personas, n_components) projected coordinates.
        explained_ratio: (n_components,) explained variance ratio.
    """
    if not persona_vectors:
        raise ValueError("persona_vectors is empty")

    names = sorted(persona_vectors.keys())
    matrix = torch.stack([persona_vectors[name][layer].float() for name in names], dim=0)
    if center:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)

    q = min(n_components, matrix.shape[0], matrix.shape[1])
    if q < 1:
        raise ValueError("not enough samples/features for PCA")

    u, s, _ = torch.pca_lowrank(matrix, q=q)
    coords = u[:, :q] * s[:q]
    variances = (s[:q] ** 2) / max(matrix.shape[0] - 1, 1)
    total_var = matrix.var(dim=0, unbiased=True).sum().clamp_min(1e-12)
    explained_ratio = variances / total_var
    return names, coords, explained_ratio


def save_projection_artifact(
    names: list[str],
    coords: torch.Tensor,
    explained_ratio: torch.Tensor,
    filename: str,
) -> None:
    """Save PCA/UMAP-ready projection artifact as JSON in artifacts/plots."""
    if coords.ndim != 2:
        raise ValueError("coords must have shape (n_personas, n_components)")
    if len(names) != coords.shape[0]:
        raise ValueError("names length must match coords row count")

    output = {
        "names": names,
        "coords": coords.tolist(),
        "explained_variance_ratio": explained_ratio.tolist(),
    }
    output_path = _plots_dir() / f"{filename}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Projection artifact saved to {output_path}")
