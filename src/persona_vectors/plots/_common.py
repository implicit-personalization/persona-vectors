"""Shared building blocks for the plotting package.

IO helpers, the layer slider/animation controls used by every layered
figure, and small validation utilities. Anything that more than one
plotting submodule needs lives here.
"""

from pathlib import Path

import plotly.graph_objects as go
import torch
from persona_data.environment import get_artifacts_dir
from plotly.colors import qualitative


def _plots_dir() -> Path:
    path = get_artifacts_dir() / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_plot_html(fig: go.Figure, filename: str) -> Path:
    """Save a Plotly figure as an HTML artifact."""
    output_path = _plots_dir() / f"{filename}.html"
    fig.write_html(str(output_path))
    return output_path


def finalize(fig: go.Figure, filename: str | None, show: bool) -> None:
    if filename is not None:
        output_path = save_plot_html(fig, filename)
        print(f"Plot saved to {output_path}")
    if show:
        fig.show()


# Shared figure typography so every plot's text reads at the same size.
# Tuned to match the probe metric-comparison figure.
TITLE_FONT_SIZE = 24
BASE_FONT_SIZE = 18
AXIS_TITLE_FONT_SIZE = 20
TICK_FONT_SIZE = 16

# NOTE: To download figure plots for presentation etc
# TITLE_FONT_SIZE = 45
# BASE_FONT_SIZE = 28
# AXIS_TITLE_FONT_SIZE = 25
# TICK_FONT_SIZE = 18


def apply_fig_fonts(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the shared font sizes so every plot's text reads at one size.

    Fonts only — axis ranges, legends, templates and traces are left
    untouched, so this is safe to call on line plots, heatmaps and
    dendrograms alike. Pass ``title`` to also set the title text.
    """
    title_kw: dict = {"font": {"size": TITLE_FONT_SIZE}}
    if title is not None:
        title_kw["text"] = title
    fig.update_layout(font=dict(size=BASE_FONT_SIZE), title=title_kw)
    fig.update_xaxes(
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )
    fig.update_yaxes(
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )
    # 3D scatter plots use scene axes, not cartesian ones; no-op otherwise.
    fig.update_scenes(
        xaxis_title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        yaxis_title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        zaxis_title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        xaxis_tickfont=dict(size=TICK_FONT_SIZE),
        yaxis_tickfont=dict(size=TICK_FONT_SIZE),
        zaxis_tickfont=dict(size=TICK_FONT_SIZE),
    )
    return fig


def label_color_map(labels: list[str]) -> dict[str, str]:
    palette = qualitative.Safe + qualitative.Dark24 + qualitative.Set3
    unique_labels = sorted(set(labels), key=lambda value: value.casefold())
    return {
        label: palette[index % len(palette)]
        for index, label in enumerate(unique_labels)
    }


def validate_layers(vectors: torch.Tensor, layers: list[int] | None) -> list[int]:
    num_layers = int(vectors.shape[1])
    selected = list(range(num_layers)) if layers is None else list(layers)
    invalid = [layer for layer in selected if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"Invalid layer(s) for tensor with {num_layers} layers: {invalid}"
        )
    return selected


def layer_slider(selected_layers: list[int], pad_t: int = 45) -> list[dict]:
    return [
        dict(
            active=0,
            currentvalue=dict(prefix="Layer "),
            pad=dict(t=pad_t),
            steps=[
                dict(
                    label=str(layer),
                    method="animate",
                    args=[
                        [str(layer)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                )
                for layer in selected_layers
            ],
        )
    ]


def layer_animation_buttons() -> list[dict]:
    return [
        dict(
            type="buttons",
            direction="left",
            active=-1,
            x=1,
            xanchor="right",
            y=1.16,
            yanchor="top",
            bgcolor="#f8fafc",
            bordercolor="#94a3b8",
            font=dict(color="#111827", size=13),
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=650, redraw=True),
                            transition=dict(duration=250),
                            fromcurrent=True,
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        )
    ]


def layer_frame_layout(
    title: str,
    layer: int,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    z_range: list[float] | None = None,
) -> dict:
    layout = {
        "title": {
            "text": f"{title} - Layer {layer}",
            "font": {"size": TITLE_FONT_SIZE},
            "y": 0.98,
            "yanchor": "top",
        }
    }
    if z_range is not None:
        scene: dict = {}
        if x_range is not None:
            scene["xaxis"] = {"range": x_range}
        if y_range is not None:
            scene["yaxis"] = {"range": y_range}
        scene["zaxis"] = {"range": z_range}
        layout["scene"] = scene
        return layout
    if x_range is not None:
        layout["xaxis"] = {"range": x_range}
    if y_range is not None:
        layout["yaxis"] = {"range": y_range}
    return layout


def coordinate_range(coords: torch.Tensor, axis: int) -> list[float]:
    values = coords[:, axis].float().cpu()
    low = float(values.min())
    high = float(values.max())
    if low == high:
        padding = 1.0
    else:
        padding = (high - low) * 0.08
    return [low - padding, high + padding]
