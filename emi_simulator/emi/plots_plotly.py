"""Plotly figures: the animated virtual lab and the interactive live graphs."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import SWEEP_INFO, coil_face_toward_magnet, current_sense_from_left, fmt_si
from .scene import CM, SceneGeometry, current_symbols, frame_indices

PALETTES = {
    "light": {"bg": "#FFFFFF", "panel": "#F3F5F8", "ink": "#1D2733", "muted": "#5C6B7A",
              "grid": "#E3E8EE", "field": "#8C9BAB", "coil": "#B87333",
              "coil_back": "rgba(184,115,51,0.30)", "N": "#C8323C", "S": "#2F5DA8",
              "flux": "#6B4FBB", "flux_neg": "#1F8FB8", "emf": "#0F8B6E", "current": "#C26A00",
              "needle": "#B3261E", "lcd": "#E6EFE9", "lcd_ink": "#14382B"},
    "dark": {"bg": "#121A23", "panel": "#1A2430", "ink": "#E6EDF3", "muted": "#9AA8B6",
             "grid": "#2A3746", "field": "#6F8396", "coil": "#D08A4A",
             "coil_back": "rgba(208,138,74,0.30)", "N": "#E0525B", "S": "#5B8BD9",
             "flux": "#A48BF0", "flux_neg": "#4FC1E9", "emf": "#3CC3A0", "current": "#F0A030",
             "needle": "#FF6B5E", "lcd": "#0D2A20", "lcd_ink": "#9FF0CF"},
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _nice_ceil(x: float) -> float:
    """Round up to 1, 2 or 5 × 10^k (for meter full-scale values)."""
    if x <= 0:
        return 1.0
    e = np.floor(np.log10(x))
    for f in (1, 2, 5, 10):
        if f * 10**e >= x * 0.999:
            return float(f * 10**e)
    return float(10 ** (e + 1))


def _base_layout(fig, pal, height):
    fig.update_layout(
        height=height, paper_bgcolor=pal["bg"], plot_bgcolor=pal["bg"],
        font=dict(color=pal["ink"], family="Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=12),
        margin=dict(l=60, r=30, t=60, b=50), showlegend=False,
    )
    fig.update_xaxes(gridcolor=pal["grid"], zerolinecolor=pal["muted"], linecolor=pal["muted"])
    fig.update_yaxes(gridcolor=pal["grid"], zerolinecolor=pal["muted"], linecolor=pal["muted"])


# ---------------------------------------------------------------------------
# Animated virtual laboratory
# ---------------------------------------------------------------------------
def lab_animation(res, theme: str = "light", n_frames: int = 90, height: int = 760):
    """Side view + instruments + three time graphs, animated with Plotly frames.

    Returns (figure, playback_seconds).
    """
    pal = PALETTES[theme]
    p = res.params
    g = SceneGeometry.from_result(res)
    i_peak = int(np.argmax(np.abs(res.emf)))
    idx = frame_indices(len(res.t), n_frames, must_include=[i_peak])
    I_max = float(np.abs(res.current).max())
    I_fs = _nice_ceil(I_max) if I_max > 0 else 1.0
    I_thr = 1e-3 * I_max if I_max > 0 else np.inf
    flux_max = float(np.abs(res.flux).max()) or 1.0

    fig = make_subplots(
        rows=2, cols=3, specs=[[{"colspan": 2}, None, {}], [{}, {}, {}]],
        row_heights=[0.62, 0.38], column_widths=[0.34, 0.33, 0.33],
        horizontal_spacing=0.07, vertical_spacing=0.13,
        subplot_titles=("Side view (cm) — magnet moves along the x-axis", "Meters",
                        "Flux per turn Φ (µWb)", "Induced EMF ε (V)", "Induced current I (mA)"))

    def add(trace, row, col):
        fig.add_trace(trace, row=row, col=col)
        return len(fig.data) - 1

    # ---------------- static: coil, normal ----------------
    front, back, drawn = g.coil_turns(p.N)
    add(go.Scatter(x=back[0], y=back[1], mode="lines", line=dict(color=pal["coil_back"], width=1.5),
                   hoverinfo="skip"), 1, 1)
    add(go.Scatter(x=front[0], y=front[1], mode="lines", line=dict(color=pal["coil"], width=2.2),
                   hoverinfo="skip"), 1, 1)
    nx, ny = g.normal_arrow()
    add(go.Scatter(x=nx, y=ny, mode="lines+markers+text", line=dict(color=pal["muted"], dash="dash", width=1.2),
                   marker=dict(size=[0, 10], symbol="arrow", angleref="previous", color=pal["muted"]),
                   text=["", "  n̂"], textposition="middle right", hoverinfo="skip"), 1, 1)
    add(go.Scatter(x=[g.xlim[0] + 0.5], y=[g.ylim[0] + 0.06 * (g.ylim[1] - g.ylim[0])], mode="text",
                   text=[f"Coil: N = {p.N} turns ({drawn} drawn), a = {g.a:.2f} cm, θ = {p.theta_deg:g}°.  "
                         f"⊙ current out of page, ⊗ into page"],
                   textposition="middle right", textfont=dict(size=10, color=pal["muted"]),
                   hoverinfo="skip"), 1, 1)

    # ---------------- static: instruments ----------------
    gc = np.array([0.0, 0.10])
    Rg = 0.95
    arc = np.deg2rad(np.linspace(30, 150, 80))
    add(go.Scatter(x=gc[0] + Rg * np.cos(arc), y=gc[1] + Rg * np.sin(arc), mode="lines",
                   line=dict(color=pal["ink"], width=2), hoverinfo="skip"), 1, 3)
    tick_x, tick_y = [], []
    for f in np.linspace(-1, 1, 9):
        ang = np.deg2rad(90 - 60 * f)
        r0 = 0.86 if f in (-1, 0, 1) else 0.90
        tick_x += [gc[0] + r0 * Rg * np.cos(ang), gc[0] + Rg * np.cos(ang), None]
        tick_y += [gc[1] + r0 * Rg * np.sin(ang), gc[1] + Rg * np.sin(ang), None]
    add(go.Scatter(x=tick_x, y=tick_y, mode="lines", line=dict(color=pal["ink"], width=1.5),
                   hoverinfo="skip"), 1, 3)
    lab_ang = np.deg2rad([150, 90, 30])
    add(go.Scatter(x=gc[0] + 1.13 * Rg * np.cos(lab_ang), y=gc[1] + 1.10 * Rg * np.sin(lab_ang),
                   mode="text", text=[f"−{fmt_si(I_fs, 'A', 3)}", "0", f"+{fmt_si(I_fs, 'A', 3)}"],
                   textfont=dict(size=10, color=pal["muted"]), hoverinfo="skip"), 1, 3)
    add(go.Scatter(x=[0], y=[gc[1] + 0.35], mode="text", text=["G (centre-zero)"],
                   textfont=dict(size=10, color=pal["muted"]), hoverinfo="skip"), 1, 3)
    # voltmeter LCD box
    bx = [-1.05, 1.05, 1.05, -1.05, -1.05]
    by = [-0.68, -0.68, -0.08, -0.08, -0.68]
    add(go.Scatter(x=bx, y=by, mode="lines", fill="toself", fillcolor=pal["lcd"],
                   line=dict(color=pal["muted"], width=1), hoverinfo="skip"), 1, 3)
    # front-view inset
    ic, ir = np.array([0.0, -1.32]), 0.36
    s = np.linspace(0, 2 * np.pi, 100)
    add(go.Scatter(x=ic[0] + ir * np.cos(s), y=ic[1] + ir * np.sin(s), mode="lines",
                   line=dict(color=pal["coil"], width=3), hoverinfo="skip"), 1, 3)
    add(go.Scatter(x=[0], y=[ic[1] + ir + 0.15], mode="text", text=["Coil seen from the left"],
                   textfont=dict(size=10, color=pal["muted"]), hoverinfo="skip"), 1, 3)

    # ---------------- static: full time graphs ----------------
    add(go.Scatter(x=res.t, y=res.flux * 1e6, mode="lines", line=dict(color=pal["flux"], width=2),
                   name="Φ", hovertemplate="t=%{x:.4f} s<br>Φ=%{y:.4g} µWb<extra></extra>"), 2, 1)
    add(go.Scatter(x=res.t, y=res.emf, mode="lines", line=dict(color=pal["emf"], width=2),
                   name="ε", hovertemplate="t=%{x:.4f} s<br>ε=%{y:.4g} V<extra></extra>"), 2, 2)
    add(go.Scatter(x=res.t, y=res.current * 1e3, mode="lines", line=dict(color=pal["current"], width=2),
                   name="I", hovertemplate="t=%{x:.4f} s<br>I=%{y:.4g} mA<extra></extra>"), 2, 3)

    # ---------------- dynamic traces ----------------
    def frame_state(k: int) -> list[go.Scatter]:
        X = res.x[k] * CM
        I = float(res.current[k])
        phi = float(res.flux[k])
        emf = float(res.emf[k])
        halves = g.magnet_halves(X)
        fx, fy, fs = g.field_lines(X, n_pts=90)
        fx, fy = np.round(fx, 3), np.round(fy, 3)
        syms, on = current_symbols(I, I_thr)
        size_c = (10 + 12 * abs(I) / I_max) * on if I_max > 0 else 0
        alpha = 0.15 + 0.8 * abs(phi) / flux_max
        seg_col = _rgba(pal["flux"] if phi >= 0 else pal["flux_neg"], alpha)
        f = float(np.clip(I / I_fs, -1.05, 1.05))
        ang = np.deg2rad(90 - 60 * f)
        sense = current_sense_from_left(I, p.theta_deg) if on else "none"
        face = coil_face_toward_magnet(I, res.x[k], p.theta_deg) if on else "–"
        # induced poles on the two faces of the coil
        plus_face, minus_face = g.face_label_points()
        if on:
            plus_lab, minus_lab = ("N", "S") if I > 0 else ("S", "N")
        else:
            plus_lab, minus_lab = "", ""
        # front-view arrows
        alphas = np.deg2rad([45, 135, 225, 315])
        if sense == "anticlockwise":
            angs = [-np.rad2deg(a_) for a_ in alphas]
        elif sense == "clockwise":
            angs = [180 - np.rad2deg(a_) for a_ in alphas]
        else:
            angs = [0, 0, 0, 0]
        arrow_size = 16 if sense != "none" else 0
        status = {"none": "No current", "clockwise": "Current: clockwise",
                  "anticlockwise": "Current: anticlockwise"}[sense]
        if face != "–":
            status += f"<br>Face toward magnet: {face} ({'repels' if face == p.polarity else 'attracts'})"
        t = float(res.t[k])

        def cursor(y_full, scale):
            lo, hi = float(np.min(y_full)) * scale, float(np.max(y_full)) * scale
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            return go.Scatter(x=[t, t], y=[lo - pad, hi + pad])

        return [
            go.Scatter(x=fx, y=fy),                                             # field lines
            go.Scatter(x=halves["S"][0], y=halves["S"][1]),                     # magnet S
            go.Scatter(x=halves["N"][0], y=halves["N"][1]),                     # magnet N
            go.Scatter(x=[halves["N_label"][0], halves["S_label"][0]], y=[0, 0]),  # pole letters
            go.Scatter(line=dict(color=seg_col, width=7)),                      # flux "disc"
            go.Scatter(marker=dict(symbol=syms, size=[size_c, size_c])),        # ⊙ ⊗
            go.Scatter(x=[plus_face[0], minus_face[0]], y=[plus_face[1], minus_face[1]],
                       text=[plus_lab, minus_lab]),                             # induced poles
            go.Scatter(x=[gc[0], gc[0] + 0.85 * Rg * np.cos(ang)],
                       y=[gc[1], gc[1] + 0.85 * Rg * np.sin(ang)]),             # needle
            go.Scatter(text=[f"ε = {fmt_si(emf, 'V', 4)}", f"I = {fmt_si(I, 'A', 4)}",
                             f"Φ = {fmt_si(phi, 'Wb', 4)}"]),                   # LCD readings
            go.Scatter(marker=dict(angle=angs, size=arrow_size)),               # inset arrows
            go.Scatter(text=[status]),                                          # status text
            go.Scatter(text=[f"t = {t:.4f} s   magnet centre x = {X:+.1f} cm"]),  # clock
            cursor(res.flux, 1e6), go.Scatter(x=[t], y=[phi * 1e6]),
            cursor(res.emf, 1.0), go.Scatter(x=[t], y=[emf]),
            cursor(res.current, 1e3), go.Scatter(x=[t], y=[I * 1e3]),
        ]

    k0 = int(idx[0])
    st0 = frame_state(k0)
    seg_x, seg_y = g.flux_segment()
    cx, cy = g.current_marker_points()
    dyn = []
    sizes0 = g.field_lines(res.x[k0] * CM, n_pts=90)[2]
    dyn.append(add(go.Scatter(x=st0[0].x, y=st0[0].y, mode="lines+markers",
                              line=dict(color=pal["field"], width=1),
                              marker=dict(size=sizes0, symbol="arrow", angleref="previous",
                                          color=pal["field"]), hoverinfo="skip"), 1, 1))
    dyn.append(add(go.Scatter(x=st0[1].x, y=st0[1].y, mode="lines", fill="toself", fillcolor=pal["S"],
                              line=dict(color=pal["ink"], width=1), hoverinfo="skip"), 1, 1))
    dyn.append(add(go.Scatter(x=st0[2].x, y=st0[2].y, mode="lines", fill="toself", fillcolor=pal["N"],
                              line=dict(color=pal["ink"], width=1), hoverinfo="skip"), 1, 1))
    dyn.append(add(go.Scatter(x=st0[3].x, y=st0[3].y, mode="text", text=["N", "S"],
                              textfont=dict(color="white", size=13, family="Arial Black, Arial"),
                              hoverinfo="skip"), 1, 1))
    dyn.append(add(go.Scatter(x=seg_x, y=seg_y, mode="lines", line=st0[4].line, hoverinfo="skip"), 1, 1))
    dyn.append(add(go.Scatter(x=cx, y=cy, mode="markers",
                              marker=dict(symbol=st0[5].marker.symbol, size=st0[5].marker.size,
                                          color=pal["current"], line=dict(width=2, color=pal["current"])),
                              hoverinfo="skip"), 1, 1))
    dyn.append(add(go.Scatter(x=st0[6].x, y=st0[6].y, mode="text", text=st0[6].text,
                              textfont=dict(size=12, color=pal["muted"]), hoverinfo="skip"), 1, 1))
    dyn.append(add(go.Scatter(x=st0[7].x, y=st0[7].y, mode="lines", line=dict(color=pal["needle"], width=3),
                              hoverinfo="skip"), 1, 3))
    dyn.append(add(go.Scatter(x=[0, 0, 0], y=[-0.24, -0.42, -0.58], mode="text", text=st0[8].text,
                              textfont=dict(size=[15, 12, 11], color=pal["lcd_ink"],
                                            family="Consolas, Menlo, monospace"),
                              hoverinfo="skip"), 1, 3))
    ang_pts = np.deg2rad([45, 135, 225, 315])
    dyn.append(add(go.Scatter(x=ic[0] + ir * np.cos(ang_pts), y=ic[1] + ir * np.sin(ang_pts), mode="markers",
                              marker=dict(symbol="arrow", angle=st0[9].marker.angle, size=st0[9].marker.size,
                                          color=pal["current"]), hoverinfo="skip"), 1, 3))
    dyn.append(add(go.Scatter(x=[0], y=[-1.88], mode="text", text=st0[10].text,
                              textfont=dict(size=10, color=pal["ink"]), hoverinfo="skip"), 1, 3))
    dyn.append(add(go.Scatter(x=[g.xlim[0] + 0.5], y=[g.ylim[1] - 0.06 * (g.ylim[1] - g.ylim[0])],
                              mode="text", text=st0[11].text, textposition="middle right",
                              textfont=dict(size=12, color=pal["ink"]), hoverinfo="skip"), 1, 1))
    for (row, col), color, j in (((2, 1), pal["flux"], 12), ((2, 2), pal["emf"], 14), ((2, 3), pal["current"], 16)):
        dyn.append(add(go.Scatter(x=st0[j].x, y=st0[j].y, mode="lines",
                                  line=dict(color=pal["muted"], width=1, dash="dot"), hoverinfo="skip"), row, col))
        dyn.append(add(go.Scatter(x=st0[j + 1].x, y=st0[j + 1].y, mode="markers",
                                  marker=dict(size=10, color=color, line=dict(width=2, color=pal["bg"])),
                                  hoverinfo="skip"), row, col))

    # ---------------- frames, slider, buttons ----------------
    frames = []
    for k in idx:
        frames.append(go.Frame(data=frame_state(int(k)), traces=dyn, name=f"f{int(k)}"))
    fig.frames = frames

    playback = float(np.clip(res.duration * 8.0, 4.0, 14.0))
    frame_ms = max(30, int(1000 * playback / len(idx)))
    play_args = dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0),
                     fromcurrent=True, mode="immediate")
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.06, xanchor="left", yanchor="top",
                          pad=dict(r=6, t=0), showactive=False, bgcolor=pal["panel"],
                          font=dict(color=pal["ink"]),
                          buttons=[
                              dict(label="▶ Play", method="animate", args=[None, play_args]),
                              dict(label="❚❚ Pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                      transition=dict(duration=0), mode="immediate")]),
                              dict(label="⟲ Replay", method="animate",
                                   args=[None, dict(play_args, fromcurrent=False)]),
                          ])],
        sliders=[dict(active=0, x=0.22, y=-0.04, len=0.78, xanchor="left", yanchor="top",
                      pad=dict(t=0, b=0), font=dict(color="rgba(0,0,0,0)", size=8),
                      ticklen=0, minorticklen=0, bgcolor=pal["panel"], bordercolor=pal["muted"],
                      currentvalue=dict(prefix="t = ", suffix=" s", font=dict(size=11, color=pal["ink"])),
                      steps=[dict(method="animate", label=f"{res.t[k]:.3f}",
                                  args=[[f"f{int(k)}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                             transition=dict(duration=0))])
                             for k in idx])],
    )
    _base_layout(fig, pal, height)
    fig.update_layout(margin=dict(l=50, r=20, t=50, b=95))
    # axes
    fig.update_xaxes(range=list(g.xlim), row=1, col=1, title_text="x (cm)")
    fig.update_yaxes(range=list(g.ylim), row=1, col=1, scaleanchor="x", scaleratio=1, title_text="y (cm)")
    fig.update_xaxes(range=[-1.25, 1.25], visible=False, row=1, col=3)
    fig.update_yaxes(range=[-2.0, 1.2], visible=False, row=1, col=3, scaleanchor="x2", scaleratio=1)
    for col, lab in ((1, "Φ (µWb)"), (2, "ε (V)"), (3, "I (mA)")):
        fig.update_xaxes(title_text="t (s)", row=2, col=col)
        fig.update_yaxes(title_text=lab, row=2, col=col)
    return fig, playback


# ---------------------------------------------------------------------------
# Live graphs
# ---------------------------------------------------------------------------
def _graph(pal, height=340):
    fig = go.Figure()
    _base_layout(fig, pal, height)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                      margin=dict(l=60, r=30, t=70, b=50))
    return fig


def emf_vs_time(res, theme="light"):
    pal = PALETTES[theme]
    fig = _graph(pal)
    fig.add_trace(go.Scatter(x=res.t, y=res.emf, mode="lines", name="ε(t) = −N dΦ/dt",
                             line=dict(color=pal["emf"], width=2.2),
                             hovertemplate="t = %{x:.4f} s<br>ε = %{y:.5g} V<extra></extra>"))
    m = res.metrics()
    fig.add_hline(y=0, line=dict(color=pal["muted"], width=1))
    if m["max_abs_emf_V"] > 0:
        i = int(np.argmax(np.abs(res.emf)))
        fig.add_trace(go.Scatter(x=[res.t[i]], y=[res.emf[i]], mode="markers+text", name="peak",
                                 text=[f" peak {fmt_si(res.emf[i], 'V')}"], textposition="middle right",
                                 marker=dict(size=9, color=pal["emf"])))
    fig.update_layout(title="1. Induced EMF vs time")
    fig.update_xaxes(title_text="Time t (s)")
    fig.update_yaxes(title_text="Induced EMF ε (V)")
    return fig


def flux_vs_time(res, theme="light"):
    pal = PALETTES[theme]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _base_layout(fig, pal, 340)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                      margin=dict(l=60, r=60, t=70, b=50), title="5. Magnetic flux and its rate of change vs time")
    fig.add_trace(go.Scatter(x=res.t, y=res.flux * 1e6, name="Φ per turn (µWb)", mode="lines",
                             line=dict(color=pal["flux"], width=2.2),
                             hovertemplate="t = %{x:.4f} s<br>Φ = %{y:.5g} µWb<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=res.t, y=res.dflux_dt * 1e3, name="dΦ/dt (mWb/s)", mode="lines",
                             line=dict(color=pal["muted"], width=1.4, dash="dash"),
                             hovertemplate="t = %{x:.4f} s<br>dΦ/dt = %{y:.5g} mWb/s<extra></extra>"), secondary_y=True)
    fig.update_xaxes(title_text="Time t (s)")
    fig.update_yaxes(title_text="Flux per turn Φ (µWb)", secondary_y=False)
    fig.update_yaxes(title_text="dΦ/dt (mWb/s)", secondary_y=True, showgrid=False)
    return fig


def current_vs_time(res, theme="light"):
    pal = PALETTES[theme]
    fig = _graph(pal)
    I = res.current * 1e3
    fig.add_trace(go.Scatter(x=res.t, y=np.where(I >= 0, I, 0), mode="lines", name="positive (right-handed about n̂)",
                             line=dict(color=pal["current"], width=0), fill="tozeroy",
                             fillcolor=_rgba(pal["current"], 0.25), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=res.t, y=np.where(I < 0, I, 0), mode="lines", name="negative",
                             line=dict(color=pal["S"], width=0), fill="tozeroy",
                             fillcolor=_rgba(pal["S"], 0.20), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=res.t, y=I, mode="lines", name="I(t) = ε/R",
                             line=dict(color=pal["current"], width=2.2),
                             hovertemplate="t = %{x:.4f} s<br>I = %{y:.5g} mA<extra></extra>"))
    fig.update_layout(title=f"6. Induced current vs time (R = {res.params.R:g} Ω)")
    fig.update_xaxes(title_text="Time t (s)")
    fig.update_yaxes(title_text="Induced current I (mA)")
    return fig


def sweep_figure(sw: dict, current_value=None, theme="light", number: str = ""):
    pal = PALETTES[theme]
    info = SWEEP_INFO.get(sw["name"], {"label": sw["name"], "unit": "", "symbol": sw["name"], "held": ""})
    fig = _graph(pal)
    fit = sw["fit"]
    if np.isfinite(fit["k"]) and sw["name"] in ("B_ref", "N", "v"):
        xx = np.linspace(0, float(np.max(sw["x"])) * 1.05, 60)
        fig.add_trace(go.Scatter(x=xx, y=fit["k"] * xx, mode="lines", name=f"ε ∝ {info['symbol']} (R² = {fit['r2']:.5f})",
                                 line=dict(color=pal["muted"], dash="dash", width=1.3)))
    fig.add_trace(go.Scatter(x=sw["x"], y=sw["peak"], mode="markers+lines", name="simulated peak |ε|",
                             line=dict(color=pal["emf"], width=1), marker=dict(size=8, color=pal["emf"]),
                             hovertemplate=f"{info['symbol']} = %{{x:.4g}} {info['unit']}<br>peak |ε| = %{{y:.5g}} V<extra></extra>"))
    if current_value is not None:
        fig.add_vline(x=current_value, line=dict(color=pal["N"], dash="dot", width=1.2),
                      annotation_text="current setting", annotation_font_color=pal["N"])
    title = f"{number + '. ' if number else ''}Peak induced EMF vs {info['label']}"
    fig.update_layout(title=dict(text=f"{title}<br><sup>Held constant: {info['held']}</sup>"))
    fig.update_xaxes(title_text=f"{info['label']} ({info['unit']})", rangemode="tozero")
    fig.update_yaxes(title_text="Peak |ε| (V)", rangemode="tozero")
    return fig


def lenz_figure(lenz: dict, theme="light"):
    pal = PALETTES[theme]
    fig = _graph(pal, 380)
    styles = {"N pole approaches": (pal["N"], "solid"), "N pole moves away": (pal["N"], "dash"),
              "S pole approaches": (pal["S"], "solid"), "S pole moves away": (pal["S"], "dash")}
    for label, res in lenz["results"].items():
        c, d = styles[label]
        fig.add_trace(go.Scatter(x=res.t / res.t[-1], y=res.emf, mode="lines", name=label,
                                 line=dict(color=c, dash=d, width=2)))
    eo = lenz["edge_on"]
    fig.add_trace(go.Scatter(x=eo.t / eo.t[-1], y=eo.emf, mode="lines", name="N approaches, coil edge-on (θ = 90°)",
                             line=dict(color=pal["muted"], dash="dot", width=2)))
    fig.update_layout(title="Lenz's law: EMF for the four basic cases", legend=dict(orientation="v", x=1.02, y=1,
                                                                                     xanchor="left", yanchor="top"))
    fig.update_xaxes(title_text="Normalised time t / T")
    fig.update_yaxes(title_text="Induced EMF ε (V)")
    return fig
