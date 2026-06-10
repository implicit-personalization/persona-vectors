"""Attribute co-occurrence heatmap (Cramér's V over the persona population)."""

import numpy as np
import plotly.graph_objects as go

from persona_vectors.plots._common import apply_fig_fonts, finalize


def build_cooccurrence_heatmap(
    labels: list[str],
    matrix: np.ndarray,
    title: str = "Attribute co-occurrence (Cramér's V)",
    filename: str | None = None,
    show: bool = False,
    cell_px: int = 44,
) -> go.Figure:
    """Heatmap of a symmetric attribute association matrix in ``[0, 1]``.

    ``matrix`` is the ``(A, A)`` array from
    :func:`persona_vectors.correlations.attribute_association_matrix`; ``labels``
    are the matching attribute names. Mirrors the similarity-heatmap styling but
    uses a sequential 0..1 scale since association is non-negative.

    The figure is sized so each cell is ``cell_px`` square (the grid is
    ``cell_px * n`` on a side); axis-label margins are added on top via
    ``automargin`` so the matrix itself stays large regardless of label length.
    """
    n = len(labels)
    grid = cell_px * n
    # Room for the (short) tick labels on the top and left; automargin grows this
    # if needed but a generous base keeps the grid square in the common case.
    label_pad = 200
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            zmin=0.0,
            zmax=1.0,
            colorscale="Blues",
            texttemplate="%{z:.2f}",
            textfont=dict(size=11),
            colorbar=dict(title="Cramér's V", thickness=18),
            hovertemplate="(%{x}, %{y})<br>Cramér's V: %{z:.4f}<extra></extra>",
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=grid + label_pad + 120,  # + colorbar
        height=grid + label_pad,
        margin=dict(t=140, b=40, l=40, r=40),
    )
    # Lock the plotting area to a square grid with equal-sized cells.
    fig.update_xaxes(side="top", tickangle=-45, automargin=True, constrain="domain")
    fig.update_yaxes(
        autorange="reversed", automargin=True, scaleanchor="x", constrain="domain"
    )
    # The default tick size re-crowds the axes as the matrix grows, so scale down.
    apply_fig_fonts(fig, tick_size=15 if n <= 16 else 12 if n <= 28 else 10)
    finalize(fig, filename, show)
    return fig
