#!/usr/bin/env python
"""Polished single-file report for supervisor: deconfounded trait vectors + steering.

Reads every artifact from ``artifacts/trait_report/`` — the foundation/band
matrices (``gemma_*.npz/json``, ``multilayer_diag``, ``ood_geometry``) plus the
experiment JSONs (``generalization``, ``copyhead``, ``merge``, ``adaptive``,
``perpos``). Emits one self-contained HTML with embedded plotly.
"""

import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from persona_vectors.correlations import (
    matrix_permutation_test,
    matrix_spearman,
    rank_delta_matrix,
)

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "trait_report"

OUT = ROOT / "report_final.html"


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def esc(s):
    return html.escape(str(s))


_first = [True]


def figdiv(fig):
    d = pio.to_html(
        fig,
        include_plotlyjs=("inline" if _first[0] else False),
        full_html=False,
        config={"displayModeBar": False},
    )
    _first[0] = False
    return f"<div class='fig'>{d}</div>"


def heat(z, labels, title, diverging=False, zlabel=""):
    kw = (
        dict(colorscale="RdBu", zmid=0, zmin=-1, zmax=1)
        if diverging
        else dict(colorscale="Blues", zmin=0, zmax=1)
    )
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            texttemplate="%{z:.2f}",
            textfont=dict(size=9),
            colorbar=dict(title=zlabel, thickness=14),
            xgap=1,
            ygap=1,
            **kw,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=820,
        height=720,
        margin=dict(t=110, b=30, l=30, r=30),
    )
    fig.update_xaxes(side="top", tickangle=-45, automargin=True)
    fig.update_yaxes(autorange="reversed", automargin=True, scaleanchor="x")
    return fig


# ---------- load ----------
_matrices = ROOT / "gemma_matrices.npz"
if not _matrices.exists():
    raise SystemExit(
        f"missing required artifact {_matrices} — run the trait-report analysis "
        "to populate artifacts/trait_report/ first"
    )
mats = np.load(_matrices, allow_pickle=True)
attrs = list(mats["attrs"])
cos, co = mats["cos"], mats["co"]
rank_delta = rank_delta_matrix(cos, co)
found = jload(ROOT / "gemma_foundation.json")
ml = jload(ROOT / "multilayer_diag.json")
oodg = jload(ROOT / "ood_geometry.json")
gen = jload(ROOT / "generalization.json")
cp = jload(ROOT / "copyhead.json")
merge = jload(ROOT / "merge.json")
decomp = jload(ROOT / "magnitude_decomp.json")
ctrl = jload(ROOT / "merge_control.json")
ctrlflu = jload(ROOT / "merge_control_fluency.json")
ctrldir = jload(ROOT / "merge_control_dirnorm.json")
adapt = jload(ROOT / "adaptive.json")

parts = []

# ---------- header ----------
parts.append(
    "<h1>Deconfounded persona trait vectors: extraction, geometry, and steering</h1>"
    "<p class='lede'>Contrasting one persona attribute at a time with minimal-pair activation "
    "deltas yields directions that (i) are less entangled with dataset co-occurrence, and "
    "(ii) causally steer an instruction-tuned model — most effectively across a band of layers, "
    "at strengths that stay in-distribution.</p>"
    "<p class='by'>google/gemma-2-9b-it · SynthPersona (100 personas) · steering run remotely on NDIF</p>"
    "<div class='key'><div><b>Model.</b> Every result and figure below is "
    "<code>google/gemma-2-9b-it</code> (42-layer instruction-tuned), run remotely on NDIF — a single "
    "model throughout §§1–6, so cross-section comparisons are apples-to-apples.</div>"
    "<div><b>Data &amp; method.</b> SynthPersona — 100 templated synthetic personas, 14 attributes. "
    "Trait vectors are minimal-pair activation deltas with the <code>PERSONA_MEAN</code> "
    "(description-level) mask; decodability is reported at layer 21, steering uses the per-layer band "
    "L14–30. The MCQ control adds a generic-human framing (<code>PERSONA_SYS</code>) and reads option "
    "probabilities (no LLM judge).</div></div>"
    "<p class='m'>Scope note: results here are single-model. Related Gemma runs support the same "
    "qualitative geometry, but they are not plotted here. Local Llama-3.1-70B artifacts include a "
    "100-person <code>born_in_us</code> contrastive vector and several n=8 pilots; they are useful "
    "follow-ups, not a 14-attribute replication of this report.</p>"
)

# ---------- 1. method ----------
parts.append(
    "<h2>1 · Minimal-pair trait vectors</h2>"
    "<p>A population difference-of-means <em>persona</em> vector for one attribute absorbs whatever "
    "co-occurs with it. A <b>trait</b> vector controls that population confound: re-render a persona's templated "
    "description at both poles of a single attribute, extract both, and average the within-pair "
    "delta — everything that didn't change cancels.</p>"
    "<div class='eq'>v(attr) = mean over personas [ act(persona@value_to) − act(persona@value_from) ]</div>"
    "<p>Built per attribute at a mid-stack layer; oriented by value so Female→Male and Male→Female "
    "reinforce. The bundled foundation artifact reports in-sample AUC@L21 0.84–1.0 (n=100): it is a "
    "separability diagnostic, not held-out validation. The appropriate next check is to freeze these "
    "vectors and evaluate paired effects on personas outside the 100-person extraction set.</p>"
)

# ---------- 2. deconfounding (headline) ----------
parts.append("<h2>2 · Trait cosine tracks meaning, not co-occurrence</h2>")
parts.append(
    "<div class='key'><div><b>What is being compared?</b> The left matrix is a "
    "dataset statistic: Cramér's V says how strongly two persona attributes co-occur. "
    "The middle matrix is a representation statistic: |cosine| says how aligned the "
    "two extracted trait directions are.</div>"
    "<div><b>Why ranks?</b> |cosine| and Cramér's V are both bounded in [0,1], but "
    "they are not the same unit. The rank-Δ heatmap therefore compares percentile "
    "ranks within each matrix, not raw values.</div></div>"
)
parts.append(figdiv(heat(co, attrs, "Data co-occurrence (Cramér's V)", zlabel="V")))
parts.append(
    figdiv(heat(cos, attrs, "Trait-direction similarity (|cosine|)", zlabel="|cos|"))
)
parts.append(
    figdiv(
        heat(
            rank_delta,
            attrs,
            "Trait geometry rank − co-occurrence rank",
            diverging=True,
            zlabel="rank Δ",
        )
    )
)
# scatter
xs, ys, tx = [], [], []
for i in range(len(attrs)):
    for j in range(i + 1, len(attrs)):
        if np.isfinite(co[i, j]):
            xs.append(float(co[i, j]))
            ys.append(float(cos[i, j]))
            tx.append(f"{attrs[i]} · {attrs[j]}")
sf = go.Figure(
    go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        text=tx,
        marker=dict(
            size=9, color=ys, colorscale="Viridis", line=dict(width=0.5, color="#444")
        ),
        hovertemplate="%{text}<br>V=%{x:.2f} |cos|=%{y:.2f}<extra></extra>",
    )
)
sf.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#bbb"))
sf.update_layout(
    title="Alignment vs co-occurrence (points below diagonal = deconfounded)",
    xaxis_title="Cramér's V",
    yaxis_title="trait |cosine|",
    template="plotly_white",
    width=720,
    height=520,
    xaxis=dict(range=[-0.02, 1]),
    yaxis=dict(range=[-0.02, 1]),
)
parts.append(figdiv(sf))
rho, rho_p, n_pairs = matrix_spearman(cos, co)
_, perm_p = matrix_permutation_test(cos, co, n_perm=4999, seed=1337)
rank_pairs = []
for i in range(len(attrs)):
    for j in range(i + 1, len(attrs)):
        if np.isfinite(rank_delta[i, j]):
            rank_pairs.append(
                (float(rank_delta[i, j]), float(co[i, j]), float(cos[i, j]), attrs[i], attrs[j])
            )
rank_rows = []
for label, pairs in (
    ("co-occurs more than geometry aligns", sorted(rank_pairs)[:5]),
    ("geometry aligns more than co-occurrence", sorted(rank_pairs, reverse=True)[:5]),
):
    rank_rows.append(f"<tr><td class='a' colspan='5'>{label}</td></tr>")
    for d, v, c, a, b in pairs:
        rank_rows.append(
            f"<tr><td><code>{esc(a)}</code> · <code>{esc(b)}</code></td>"
            f"<td class='dl'>{d:+.2f}</td><td>{v:.2f}</td><td>{c:.2f}</td>"
            f"<td>{'candidate dataset confound' if d < 0 else 'candidate shared representation'}</td></tr>"
        )
parts.append(
    "<div class='key'><div><b>Spurious co-occurrence removed.</b> <code>born_in_us</code>·"
    "<code>same_residence_since_16</code>: V=0.38 but |cos|=0.09.</div>"
    "<div><b>Real semantics kept.</b> parents' degrees |cos|=0.82; income·wealth 0.59.</div></div>"
    "<p>Points sit below the diagonal: trait cosine reflects shared <em>concept</em>, not how often "
    "two attributes happen to travel together. The minimal pair does what it was designed to.</p>"
    f"<p class='m'>Rank summary: Spearman ρ(|cos|, V) = {rho:.2f} "
    f"(naive p = {rho_p:.3g}, matrix-permutation p = {perm_p:.3g}, {n_pairs} pairs). "
    "|cos| and V are not commensurate scales, so the heatmap above compares within-matrix "
    "percentile ranks: positive means a pair is more prominent in trait geometry than in "
    "co-occurrence; negative means the reverse.</p>"
    "<table class='t'><tr><th>attribute pair</th><th>rank Δ</th><th>V</th><th>|cos|</th>"
    f"<th>read as</th></tr>{''.join(rank_rows)}</table>"
)

# ---------- 3. decodability != steerability + band ----------
parts.append("<h2>3 · Steering: single layer is weak; a band is the fix</h2>")
parts.append(
    "<p>Every direction decodes at AUC≈1, but a single mid-layer injection barely moves generation "
    "in-distribution — decodability does not imply steerability. Steering a <b>band</b> of layers "
    "(each with its own direction and a modest per-layer strength, t≈1 = that layer's opposite-class "
    "centroid) compounds the effect while every layer stays in-distribution.</p>"
)
if ml:
    series = [
        ("baseline", "base", "#9ca3af"),
        ("single L21 (t=2)", "L21_t2", "#60a5fa"),
        ("band (t=0.5)", "band_t0.5", "#f59e0b"),
        ("band (t=1)", "band_t1", "#dc2626"),
    ]

    def nm(a, key):
        o = ml[a]
        p = np.array(o[key], float)
        p = p / p.sum()
        vals = o.get("vals")
        return (
            (float((p * np.array(vals)).sum()) - min(vals)) / (max(vals) - min(vals))
            if vals
            else float((p * np.arange(len(o["opts"]))).sum()) / (len(o["opts"]) - 1)
        )

    f = go.Figure()
    for lab, key, c in series:
        f.add_trace(
            go.Bar(name=lab, x=list(ml), y=[nm(a, key) for a in ml], marker_color=c)
        )
    f.update_layout(
        title="Single-layer vs band steering (MCQ answer toward + pole)",
        barmode="group",
        template="plotly_white",
        width=820,
        height=440,
        yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="h", y=1.13, font=dict(size=11)),
    )
    parts.append(figdiv(f))
if oodg:
    # geometry
    norm = {"age": (22, 88)}
    geo = oodg["geometry"]
    cfgs = list(next(iter(geo.values())))
    gf = go.Figure()
    for a, row in geo.items():
        ys2 = [
            (row[c] - norm[a][0]) / (norm[a][1] - norm[a][0]) if a in norm else row[c]
            for c in cfgs
        ]
        gf.add_trace(go.Scatter(x=cfgs, y=ys2, mode="lines+markers", name=a))
    gf.update_layout(
        title="Layer-band geometry (MCQ effect by band, t=1)",
        template="plotly_white",
        width=820,
        height=430,
        xaxis_title="band",
        yaxis_title="effect",
        yaxis=dict(range=[0, 1.02]),
    )
    parts.append(figdiv(gf))
    ood = oodg["ood"]
    rows = "".join(
        f"<tr><td class='c'>{esc(k)}</td><td>P(+)={v['P_no']:.2f} · rep={v['rep_frac']:.2f}</td>"
        f"<td class='g'>{esc(v['text'])}</td></tr>"
        for k, v in ood.items()
    )
    parts.append(
        "<div class='key'><div><b>Late layers (24–38) are causally dead</b>; wide contiguous bands "
        "are strongest; skipping layers hurts; the best band is attribute-specific.</div>"
        "<div><b>Band steering stays on-manifold.</b> Even t=2 keeps repeat-frac≈0 and rewrites content "
        '("a village in Italy"); the OOD case is a <em>large single layer</em>, which flips the MCQ '
        "letter but leaves the text unchanged and starts to repeat.</div></div>"
        f"<table class='t'><tr><th>config (born_in_us)</th><th>effect · fluency</th><th>generation</th></tr>{rows}</table>"
    )

# ---------- 4. merging + correlation ----------
if merge:
    parts.append(
        "<h2>4 · Merging several trait vectors, and why correlation matters</h2>"
    )
    rows = []
    for sn, d in merge.items():
        rows.append(f"<tr><td class='a' colspan='4'>{esc(sn)}</td></tr>")
        for a, r in d["per_attr"].items():
            rows.append(
                f"<tr><td>{esc(a)}</td><td>{r['unsteered']}</td><td>{r['solo']}</td>"
                f"<td class='dl'>{r['joint']}</td></tr>"
            )
    parts.append(
        "<p>Trait vectors share one residual space, so several can be added at once (summed per "
        "layer). The interesting variable is their <b>correlation</b>:</p>"
        f"<table class='t'><tr><th>trait</th><th>unsteered</th><th>solo</th><th>joint</th></tr>{''.join(rows)}</table>"
        "<div class='key'><div><b>Correlated traits reinforce.</b> <code>citizenship</code> barely "
        "steers alone (0.06) but reaches <b>0.99</b> when merged with its correlated neighbours "
        "(<code>born_in_us</code>, <code>language</code>) — co-steering compounds along the shared "
        "axis (|cos| 0.25–0.33).</div>"
        "<div><b>Orthogonal traits coexist but with cross-talk.</b> The summed perturbation grows, so "
        "joint effects differ from solo (here <code>sex</code> amplified, <code>born_in_us</code> "
        "partly suppressed). Composition is not a clean linear sum.</div></div>"
    )
    if decomp:
        rows = []
        for sn, d in decomp["sets"].items():
            rows.append(f"<tr><td class='a' colspan='3'>{esc(sn)}</td></tr>")
            for a, r in d["per_attr"].items():
                rows.append(
                    f"<tr><td>{esc(a)}</td><td>{r['raw_amp']}×</td>"
                    f"<td class='dl'>{r['proj_own_over_solo']}×</td></tr>"
                )
        b = decomp["band"]
        parts.append(
            "<p><b>Is the joint effect just larger magnitude?</b> The raw joint vector is "
            "several× longer, but what causally moves an attribute is the push along its <em>own</em> "
            f"steering direction. Decomposing the band-{b[0]}–{b[1]} joint vector onto each member's "
            "unit direction separates the two:</p>"
            f"<table class='t'><tr><th>attribute</th><th>raw ‖joint‖/‖solo‖</th>"
            f"<th>push on own axis / solo</th></tr>{''.join(rows)}</table>"
            "<div class='key'><div><b>The orthogonal cross-talk is magnitude-controlled.</b> "
            "<code>born_in_us</code> and <code>sex</code> get essentially the <em>same</em> push along "
            "their own axis in joint as solo (1.02× and 0.93×) — yet their effects swing "
            "(0.95→0.29 and 0.15→0.95). With own-axis magnitude matched, that can only be "
            "interference from the orthogonal additions, not extra magnitude.</div>"
            "<div><b>For correlated traits, joint adds magnitude on the shared axis.</b> "
            "<code>citizenship</code>'s own-axis push is <b>1.76×</b> larger in joint and its total "
            "norm <b>3.29×</b>, because the correlated neighbours (|cos| 0.03–0.29) project onto its "
            "axis. Is that magnitude what flips it, or the borrowed direction? The steering control "
            "below decides.</div></div>"
        )
    if ctrl:
        cz = ctrl["us_citizenship_status"]["p_pole"]
        parts.append(
            "<h3>Magnitude-matched control (steering run, PERSONA_SYS context)</h3>"
            "<p>The decomposition is a proxy; the generation control settles it. We steer the bare "
            "citizenship MCQ under a generic human framing (so the model isn't pinned at a pole) and "
            "compare conditions by the per-layer magnitude they put on citizenship's axis:</p>"
            "<table class='t'>"
            "<tr><th>condition</th><th>own-axis push</th><th>total norm</th><th>P(+pole)</th></tr>"
            f"<tr><td>unsteered</td><td>—</td><td>—</td><td>{cz['unsteered']}</td></tr>"
            f"<tr><td>solo t=1</td><td>1×</td><td>1×</td><td>{cz['solo_t1']}</td></tr>"
            f"<tr><td class='c'>joint t=1</td><td>1.76×</td><td>3.29×</td><td class='dl'>{cz['joint_t1']}</td></tr>"
            f"<tr><td>solo, own-axis matched</td><td>1.76×</td><td>1.76×</td><td>{cz['solo_matched_1.76']}</td></tr>"
            f"<tr><td>solo, total-norm matched</td><td>3.29×</td><td>3.29×</td><td>{cz['solo_normmatched_3.29']}</td></tr>"
            + (
                f"<tr><td class='c'>joint direction @ solo norm</td><td>—</td><td>1×</td>"
                f"<td>{ctrldir['joint_dir_at_solo_norm']['p_pole']}</td></tr>"
                if ctrldir
                else ""
            )
            + "</table>"
            "<div class='key'><div><b>The effect tracks magnitude, not direction.</b> Matching "
            f"citizenship's own-axis magnitude (1.76×) recovers only part of the joint effect "
            f"({cz['solo_matched_1.76']} vs joint {cz['joint_t1']}); matching the joint's <em>total</em> "
            f"norm (3.29×) reproduces it ({cz['solo_normmatched_3.29']}). "
            + (
                f"And the joint <em>direction</em> renormalised back to solo magnitude (1×) stays near "
                f"solo ({ctrldir['joint_dir_at_solo_norm']['p_pole']} ≈ {ctrldir['solo_t1']['p_pole']}) — "
                "so at fixed norm the correlated blend is no better than pure citizenship. "
                if ctrldir
                else ""
            )
            + "P is governed by total push magnitude in this subspace, direction-agnostic among the "
            "correlated axes.</div>"
            "<div><b>Correlation reinforces by supplying in-distribution magnitude, not a better "
            "direction.</b> The effect needs a threshold total push; co-steering reaches it with each "
            "trait at a modest, on-manifold t=1, whereas one trait alone needs 3.29× strength. The "
            "neighbours contribute magnitude, not steering know-how.</div></div>"
        )
        # figure: P(+pole) by condition, coloured by outcome (flip = magnitude, not direction)
        lab = [
            "unsteered",
            "solo t=1",
            "joint t=1",
            "solo<br>own-axis 1.76×",
            "solo<br>norm 3.29×",
        ]
        val = [
            cz["unsteered"],
            cz["solo_t1"],
            cz["joint_t1"],
            cz["solo_matched_1.76"],
            cz["solo_normmatched_3.29"],
        ]
        col = ["#9ca3af", "#9ca3af", "#dc2626", "#f59e0b", "#16a34a"]
        if ctrldir:
            lab.append("joint dir<br>@ solo norm")
            val.append(ctrldir["joint_dir_at_solo_norm"]["p_pole"])
            col.append("#9ca3af")
        cfig = go.Figure(
            go.Bar(
                x=lab,
                y=val,
                marker_color=col,
                text=[f"{v:.2f}" for v in val],
                textposition="outside",
            )
        )
        cfig.update_layout(
            title="Citizenship flip P(+pole) by condition — tracks total magnitude, not direction",
            template="plotly_white",
            width=820,
            height=440,
            yaxis=dict(range=[0, 1.08], title="P(+pole)"),
            margin=dict(t=70, b=40),
        )
        parts.append(figdiv(cfig))
        jv = ctrl["us_citizenship_status"].get("p_pole_variants", {}).get("joint_t1")
        if jv:
            parts.append(
                "<p class='m'>P(+pole) values in this section are means over three MCQ readout "
                "variants — original option order, reversed order (position-bias check), and an "
                "alternate instruction phrasing. The spread across readouts is large (citizenship "
                f"joint t=1 ranges {min(jv.values()):.2f}–{max(jv.values()):.2f}; the reversed order "
                "barely moves in any condition), so single-readout MCQ numbers overstate precision — "
                "the variant mean is the honest readout, and the magnitude-vs-direction ordering "
                "above holds under it.</p>"
            )
        if ctrlflu:
            j, s = ctrlflu["joint_t1"], ctrlflu["solo_normmatched_3.29"]
            ffig = go.Figure(
                go.Bar(
                    x=["joint (each t=1)", "solo citizenship 3.29×"],
                    y=[j["rep_frac"], s["rep_frac"]],
                    marker_color=["#dc2626", "#16a34a"],
                    text=[f"{j['rep_frac']:.3f}", f"{s['rep_frac']:.3f}"],
                    textposition="outside",
                )
            )
            ffig.update_layout(
                title="Fluency at matched total norm (repeat-frac, lower = better)",
                template="plotly_white",
                width=620,
                height=380,
                yaxis=dict(
                    range=[0, max(j["rep_frac"], s["rep_frac"]) * 1.3 + 0.02],
                    title="repeat-frac",
                ),
                margin=dict(t=70, b=40),
            )
            parts.append(figdiv(ffig))
            parts.append(
                "<p><b>And pooling stays more on-manifold.</b> Generating free text at the matched "
                f"total norm, the joint (each trait t=1) keeps repeat-frac <b>{j['rep_frac']}</b> "
                f"while concentrating the same norm on citizenship's axis (solo 3.29×) rises to "
                f"<b>{s['rep_frac']}</b> — a measurable coherence cost for reaching the flip the "
                "concentrated way. Both flip the persona (joint → "
                "<span class='m'>“originally from Argentina… I speak Spanish… picking up Catalan”</span>; "
                "solo → <span class='m'>“south of France… French… dabble in Spanish”</span>), so the "
                "effect is real but modest at this strength, not a collapse.</p>"
                "<p class='m'>Readout is MCQ option-probability (deterministic, no judge), a faithful "
                "but not bit-identical stand-in for the original free-text+judge merge metric; "
                f"born_in_us {ctrl['born_in_us']['p_pole']['unsteered']}→"
                f"{ctrl['born_in_us']['p_pole']['joint_t1']} and language "
                f"{ctrl['speak_other_language']['p_pole']['unsteered']}→"
                f"{ctrl['speak_other_language']['p_pole']['joint_t1']} reproduce the same "
                "solo&lt;joint ordering.</p>"
            )

# ---------- 5. adaptive / per-position ----------
if adapt:
    parts.append("<h2>5 · Adaptive and per-position steering</h2>")
    # rep bar
    names = ["baseline", "constant", "dim_taper", "start_third"]
    af = go.Figure()
    for a in adapt:
        af.add_trace(go.Bar(name=a, x=names, y=[adapt[a][n]["rep"] for n in names]))
    af.update_layout(
        title="Fluency under steering schedules at t=2.5 (repeat-frac, lower=better)",
        barmode="group",
        template="plotly_white",
        width=760,
        height=400,
        yaxis_title="repeated-bigram fraction",
    )
    parts.append(figdiv(af))
    ex = adapt.get("age", {})
    rows = "".join(
        f"<tr><td class='c'>{esc(n)}</td><td>rep={ex[n]['rep']:.2f}</td>"
        f"<td class='g'>{esc(ex[n]['text'])}</td></tr>"
        for n in names
        if n in ex
    )
    parts.append(
        "<p>At <em>modest</em> strength band steering is already fluent, so adaptive scheduling adds "
        "little. Its value appears when you steer <b>hard</b> (here t=2.5). For <code>age</code>, "
        "<b>constant</b> steering degenerates (repeat-frac 0.39, “called called a a man”), while the "
        "<b>Dim linear taper</b> (decrease intensity over the generation, Scalena–Sarti–Nissim Eq. 5) "
        "and the <b>per-position start-only</b> schedule (steer the first third, then release) stay "
        "fluent (0.00–0.02) and keep the trait — “we were called the Greatest Generation.”</p>"
        f"<table class='t'><tr><th>schedule (age, t=2.5)</th><th>fluency</th><th>generation</th></tr>{rows}</table>"
        "<p>So per-position / adaptive steering is the lever for pushing stubborn attributes harder "
        "without breaking fluency. (Full KL-adaptive DAC, Eq. 6, is implemented but is a per-token "
        "feedback loop — local-only — and unnecessary in the modest-strength regime.)</p>"
    )

# ---------- 6. generalization + copy heads ----------
if gen and cp:
    parts.append("<h2>6 · Generalization and the copy-head limit</h2>")
    ha = gen["heldout_actual"]
    rows = []
    for a in ["born_in_us", "age", "sex"]:
        d = gen[a]
        rows.append(
            f"<tr><td class='a' rowspan='3'>{esc(a)}<br><span class='m'>held-out={esc(ha[a])}</span></td>"
            f"<td>no prompt</td><td>{d['no_prompt']['unsteered']}</td><td>{d['no_prompt']['steered_band_t1']}</td><td class='dl'>{d['no_prompt']['delta']:+}</td></tr>"
        )
        for cn in ["generic", "heldout_persona"]:
            rows.append(
                f"<tr><td>{esc(cn)}</td><td>{d[cn]['unsteered']}</td><td>{d[cn]['steered_band_t1']}</td><td class='dl'>{d[cn]['delta']:+}</td></tr>"
            )
    cf = go.Figure()
    for cn, row in cp.items():
        ts = sorted(int(k) for k in row)
        cf.add_trace(
            go.Scatter(x=ts, y=[row[str(t)] for t in ts], mode="lines+markers", name=cn)
        )
    cf.update_layout(
        title="Copy head vs steering: P(not-US) vs band strength",
        template="plotly_white",
        width=820,
        height=440,
        xaxis_title="band strength (per layer)",
        yaxis_title="P(No)",
        legend=dict(orientation="h", y=1.13, font=dict(size=10)),
    )
    parts.append(
        "<p>Trait vectors generalize beyond their 100 extraction personas — they steer with <b>no "
        "prompt</b> and on an <b>unseen</b> held-out persona:</p>"
        f"<table class='t'><tr><th>attr</th><th>context</th><th>unsteered</th><th>steered</th><th>Δ</th></tr>{''.join(rows)}</table>"
        + figdiv(cf)
        + "<div class='key'><div><b>The limit is a copy head, not the vector.</b> When the held-out "
        "persona states a fact <em>verbatim</em> (“I was born in the United States”), an "
        "induction/copy circuit pins the answer: steering can't flip it through t=4. Delete that one "
        "sentence and the same steering flips it at t=3; the generic context flips at t=1.</div>"
        "<div><b><code>age</code> escapes</b> because its value isn't a copyable option — it steers "
        "in every context (35→55 even against the held-out persona).</div></div>"
    )

# ---------- 7. appendix: tested-but-not-adopted ----------
perpos = jload(ROOT / "perpos.json")
app = ["<h2>Appendix · What we tested and did not adopt</h2>"]
if perpos:
    app.append(
        "<h3>A. Per-position direction extraction (the paper's Δᵢ)</h3>"
        "<p>Scalena–Sarti–Nissim extract a <em>separate</em> steering direction at each generation "
        "step. Our trait vectors are pooled into one direction, so we tested whether a "
        "position-varying direction would help: from "
        f"K={perpos['K']} minimal pairs we extracted the contrast at each of the last "
        f"{perpos['n_pos']} answer positions (layer {perpos['layer']}) and measured how much those "
        "per-position directions differ from the single pooled one.</p>"
        f"<p class='m'>cos(per-position, pooled) = "
        f"{min(perpos['cos_to_pooled']):.2f}–{max(perpos['cos_to_pooled']):.2f}; "
        f"pairwise cosine between positions = {perpos['pairwise_cos_mean']:.2f} mean "
        f"(min {perpos['pairwise_cos_min']:.2f}).</p>"
        "<p><b>Verdict: not adopted.</b> The per-position directions drift, but that spread is largely "
        "noise at this K, each still tracks the pooled mean, and there is no evidence it steers "
        "better — the pooled vector across a layer band already flips attributes fluently. Per-position "
        "extraction is also a different, heavier pipeline (extract during generation), so for a "
        "<em>global</em> persona attribute it adds complexity for no gain.</p>"
    )
app.append(
    "<h3>B. Full KL-adaptive DAC vs. the simple schedule we kept</h3>"
    "<p>The paper's headline method, DAC (Eq. 6), sets the per-step intensity <em>fully "
    "automatically</em> from the KL divergence between unsteered and high-intensity distributions "
    "(capped at 2), with no user dial. We implemented it but did <b>not</b> keep it, for three "
    "reasons: (1) it removes the strength dial users want — our <code>dim_schedule</code> / "
    "<code>start_schedule</code> keep it (the schedule is a 0–1 multiplier on the user's base "
    "strength); (2) the fluency benefit DAC exists for is already delivered by the simple taper "
    "(see §5); (3) DAC is a per-token feedback loop, so it is local-only and cannot run on the remote "
    "(NDIF) path the UI uses. The simple, dial-preserving schedule is the better fit here.</p>"
)
app.append(
    "<h3>C. Method details: band steering and scheduling</h3>"
    "<p><b>Trait direction.</b> For an attribute we re-render each persona at both poles, take the "
    "within-pair residual delta per layer, and average over personas. At layer ℓ the steering vector "
    "is the unit direction <code>uℓ</code> with magnitude <code>gap_normℓ = ‖mean Δℓ‖</code>. "
    "Strength is in <em>gap units</em>: a coefficient of <code>strength · gap_normℓ</code> moves an "
    "activation by <code>strength</code> class-separations, so <code>strength=1</code> lands it at the "
    "opposite-class centroid (in-distribution).</p>"
    "<p><b>Band steering.</b> Instead of one layer we steer a contiguous band L (e.g. 14–30). The "
    "vector added at layer ℓ is <code>strength · gap_normℓ · uℓ</code> — each layer keeps its own "
    "calibrated magnitude. For several traits the per-layer vectors are <em>summed</em> "
    "(<code>vℓ = Σ_trait strength · gap_normℓ · uℓ</code>) and applied as a single <code>model.steer</code> "
    "per layer, in ascending order (multiple steers on one layer raise nnsight's "
    "<code>OutOfOrderError</code>). All band layers are steered at every generated position via "
    "<code>tracer.all()</code>.</p>"
    "<p><b>Scheduling (adaptive intensity).</b> A schedule is a multiplier "
    "<code>s(step) ∈ [0,1]</code> applied on top of the base strength, per generation step via "
    "<code>tracer.iter[step]</code>, so the layer-ℓ factor at step t is "
    "<code>s(t) · strength · gap_normℓ</code>. The user's strength dial sets the peak; the schedule "
    "only shapes it over tokens. Two shapes: the <b>linear taper</b> "
    "<code>s(t) = 1 − t/(N−1)</code> (full at the first token, 0 at the last — protects fluency once "
    "the trait is established) and <b>start-only</b> <code>s(t) = 1 if t &lt; N/3 else 0</code> "
    "(steer the opening, then release). <code>s(t)=1</code> everywhere recovers constant band steering.</p>"
)
parts.append("".join(app))

# ---------- 8. critical review ----------
_has_variants = bool(ctrl) and any(
    isinstance(v, dict) and "p_pole_variants" in v for v in ctrl.values()
)
_rev_rows = [
    (
        "Raw |cos| − V difference heatmap (§2)",
        "Cosine and Cramér's V are not commensurate scales, so subtracting raw values implies a "
        "precision the comparison doesn't have.",
        "Replaced by percentile-rank difference plus Spearman and matrix-permutation summaries. "
        "<b>Fixed in this report.</b>",
    ),
    (
        "Single-question MCQ readout (§§3–4)",
        "One question, one option order per attribute — MCQ answers are position- and wording-biased, "
        "so a flip could be an artifact of the readout.",
        "Average P(+pole) over readout variants (original / reversed option order / alternate phrasing). "
        + (
            "<b>Fixed in this report</b> — §4 values are 3-variant means."
            if _has_variants
            else "<b>Code updated</b> (scripts/merge_magnitude_control.py); re-run it to refresh §4 with "
            "variant-mean values."
        ),
    ),
    (
        "Resubstitution AUC presented as validation (§1)",
        "The old AUC uses the same minimal pairs to construct and score a direction. Bootstrap resampling "
        "does not remove that leakage, so near-ceiling values can overstate cross-persona stability.",
        "Label AUC as in-sample and, without re-fitting, score each frozen 100-person vector on a "
        "disjoint persona set. Report the paired activation effect and steering readout there.",
    ),
    (
        "What a template swap represents",
        "The deterministic v4.0 renderer preserves unrelated lines and updates only the target line, a "
        "composite line (such as age + sex), or a conditional line (religion). The direction therefore "
        "includes the attribute's intended textual realization, not an abstract feature independently of language.",
        "This is a strong controlled template intervention for removing <em>population</em> co-occurrence. "
        "Use a held-out cross-template check and a small remote minimal-pair replication to test transfer "
        "beyond that realization.",
    ),
    (
        "Hard-coded magnitude-match factors (§4)",
        "The 1.76× / 3.29× control strengths are constants copied from one decomposition run; if trait "
        "vectors are re-extracted the controls silently stop matching the joint vector.",
        "Compute both factors at runtime from the loaded bands (project the joint onto the solo unit "
        "direction per layer) instead of hard-coding. <b>Not yet changed.</b>",
    ),
    (
        "Artifact provenance",
        "Most figures here read artifacts whose generator scripts were never committed — only the §4 "
        "magnitude control (<code>merge_control*.json</code>) is regenerable today.",
        "Commit the generator scripts, or freeze the artifacts with provenance notes; tracked in "
        "<code>docs/performance.md</code>. <b>Open.</b>",
    ),
    (
        "Legacy steering calibration in code",
        "The old per-persona <code>suggested_alpha</code> (20× residual RMS) over-steers 8–28× the real "
        "class gap and collapses generation off-manifold.",
        "Superseded by gap-unit strengths (used everywhere in this report); don't use "
        "<code>suggested_alpha</code> for new work.",
    ),
    (
        "MCQ saturation (limitations)",
        "political_views / total_wealth pin at a pole under the MCQ readout, so steering there looks "
        "inert regardless of the vector.",
        "Use a free-text + judge readout for those attributes before concluding they don't steer.",
    ),
    (
        "Single-model results",
        "All plotted results are gemma-2-9b-it; band-steering conclusions could be model-specific.",
        "Replicate the §3–4 steering runs on a second model — the direction and manifold findings "
        "already replicate on gemma-3-27b, steering has not been re-run there. <b>Open.</b>",
    ),
]
parts.append(
    "<h2>Critical review · weak points and recommended changes</h2>"
    "<p>A self-review of the analyses above: what to read with caution, what has been fixed in "
    "code or in this report, and what is still open.</p>"
    "<table class='t'><tr><th>weak point</th><th>why it matters</th><th>fix / status</th></tr>"
    + "".join(
        f"<tr><td class='a'>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>"
        for r in _rev_rows
    )
    + "</table>"
)

# ---------- 9. conclusions ----------
parts.append(
    "<h2>Conclusions</h2><ul>"
    "<li><b>Minimal-pair trait vectors deconfound attributes</b> — cosine tracks meaning, not "
    "co-occurrence.</li>"
    "<li><b>Decodability ≠ steerability.</b> All directions decode at AUC≈1; causal effect is "
    "attribute- and layer-specific.</li>"
    "<li><b>Steer a band, modestly.</b> A mid-stack band at in-distribution per-layer strength flips "
    "categorical attributes (born_in_us 0.06→0.95) and stays fluent; late layers are dead, skipping "
    "hurts, large single-layer pushes go off-manifold.</li>"
    "<li><b>Merging is a magnitude effect.</b> Co-steering correlated traits reinforces "
    "(citizenship 0.06→0.99) — but the control shows it tracks total push magnitude, not the "
    "borrowed direction: one trait steered harder reproduces it, and the joint blend at solo "
    "magnitude does not. Correlation's only edge is staying on-manifold. Orthogonal traits coexist "
    "with magnitude-matched cross-talk.</li>"
    "<li><b>Adaptive / per-position steering buys headroom</b> — it preserves fluency when steering "
    "hard, where constant steering degenerates.</li>"
    "<li><b>In-context verbatim facts dominate</b> via copy heads, gating steering for stated "
    "attributes; uncopyable attributes (age) steer freely.</li>"
    "</ul>"
    "<h3>Limitations</h3><ul>"
    "<li>Synthetic templated personas; transfer to natural text untested.</li>"
    "<li>Only ordered (binary/ordinal/numeric) attributes have a single contrast direction.</li>"
    "<li>MCQ saturates for some attributes (political_views, total_wealth) — free text is the better "
    "readout there.</li>"
    "<li>Full KL-adaptive DAC is per-token (local-only); only the linear-taper / per-position "
    "variants were run remotely.</li>"
    "</ul>"
)

CSS = """
body{font:16px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,sans-serif;color:#1a1a1a;max-width:900px;margin:0 auto;padding:60px 26px 120px}
h1{font-size:36px;letter-spacing:-.02em;line-height:1.15} h2{font-size:25px;margin:56px 0 8px;border-top:1px solid #e5e7eb;padding-top:18px}
h3{font-size:18px;margin:30px 0 4px} .lede{font-size:20px;color:#374151} .by{color:#6b7280;font-size:14px}
p{margin:13px 0} code{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:14px} .m{color:#6b7280}
.eq{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:13px 16px;font-family:ui-monospace,Menlo,monospace;font-size:14px;overflow-x:auto}
.fig{margin:22px 0;display:flex;justify-content:center}
.key{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:16px 0}
.key div{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;font-size:14.5px}
table.t{width:100%;border-collapse:collapse;margin:14px 0;font-size:13.5px}
table.t td,table.t th{border-top:1px solid #e5e7eb;padding:7px 9px;text-align:left;vertical-align:top}
table.t th{font-size:11px;color:#6b7280;text-transform:uppercase}
td.a{background:#f8fafc;font-weight:600} td.c{white-space:nowrap;color:#2563eb;font-family:ui-monospace,monospace}
td.g{color:#222} td.dl{font-family:ui-monospace,monospace;color:#2563eb;font-weight:600}
ul{padding-left:22px} li{margin:6px 0}
"""
doc = (
    f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
    f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
    f"<title>Persona trait vectors & steering</title><style>{CSS}</style></head>"
    f"<body>{''.join(parts)}</body></html>"
)
OUT.write_text(doc)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")
