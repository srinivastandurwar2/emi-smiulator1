"""Static Matplotlib figures for PNG export, the HTML report and example outputs."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import SWEEP_INFO, current_sense_from_left  # noqa: E402
from .params import MOTION_MODES  # noqa: E402
from .scene import CM, SceneGeometry  # noqa: E402

COLORS = {"flux": "#6B4FBB", "emf": "#0F8B6E", "current": "#C26A00", "N": "#C8323C",
          "S": "#2F5DA8", "coil": "#B87333", "field": "#8C9BAB", "ink": "#1D2733", "fit": "#5C6B7A"}

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def _caption(res) -> str:
    p = res.params
    return (f"B₀ = {p.B_ref:g} T, N = {p.N}, v = {p.v:g} m/s, A = {p.A:g} m², θ = {p.theta_deg:g}°, "
            f"R = {p.R:g} Ω, {p.polarity} pole facing, {p.motion}")


def emf_vs_time(res):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(res.t, res.emf, color=COLORS["emf"], lw=1.8, label="ε(t) = −N dΦ/dt")
    ax.fill_between(res.t, res.emf, 0, where=res.emf > 0, color=COLORS["emf"], alpha=0.15)
    ax.fill_between(res.t, res.emf, 0, where=res.emf < 0, color=COLORS["N"], alpha=0.12)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title("Induced EMF vs time")
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Induced EMF ε (V)")
    ax.legend(loc="upper right")
    fig.text(0.01, 0.01, _caption(res), fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def flux_vs_time(res):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(res.t, res.flux * 1e6, color=COLORS["flux"], lw=1.8, label="Φ per turn")
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Flux per turn Φ (µWb)", color=COLORS["flux"])
    ax2 = ax.twinx()
    ax2.plot(res.t, res.dflux_dt * 1e3, color="#888", lw=1.1, ls="--", label="dΦ/dt")
    ax2.set_ylabel("Rate of change dΦ/dt (mWb/s)", color="#666")
    ax2.grid(False)
    ax.set_title("Magnetic flux and its rate of change vs time")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.text(0.01, 0.01, _caption(res), fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def current_vs_time(res):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(res.t, res.current * 1e3, color=COLORS["current"], lw=1.8, label="I(t) = ε(t)/R")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title("Induced current vs time (positive = right-handed about the coil normal n̂)")
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Induced current I (mA)")
    ax.legend(loc="upper right")
    fig.text(0.01, 0.01, _caption(res), fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def sweep_plot(sw: dict, current_value=None):
    info = SWEEP_INFO.get(sw["name"], {"label": sw["name"], "unit": "", "symbol": sw["name"], "held": ""})
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x, y = sw["x"], sw["peak"]
    ax.plot(x, y, "o", color=COLORS["emf"], ms=6, label="Simulated peak |ε|")
    fit = sw["fit"]
    if np.isfinite(fit["k"]) and sw["name"] in ("B_ref", "N", "v"):
        xx = np.linspace(0, x.max() * 1.05, 50)
        ax.plot(xx, fit["k"] * xx, "--", color=COLORS["fit"], lw=1.2,
                label=f"ε = k·{info['symbol']}, k = {fit['k']:.4g}, R² = {fit['r2']:.5f}")
    else:
        ax.plot(x, y, "-", color=COLORS["emf"], lw=1.0, alpha=0.6)
    if current_value is not None:
        ax.axvline(current_value, color=COLORS["N"], lw=1, ls=":", label="current setting")
    ax.set_title(f"Peak induced EMF vs {info['label']}")
    ax.set_xlabel(f"{info['label']} ({info['unit']})")
    ax.set_ylabel("Peak |ε| (V)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.01, 0.01, f"Held constant: {info['held']}", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def lenz_plot(lenz: dict):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    styles = {"N pole approaches": (COLORS["N"], "-"), "N pole moves away": (COLORS["N"], "--"),
              "S pole approaches": (COLORS["S"], "-"), "S pole moves away": (COLORS["S"], "--")}
    for label, res in lenz["results"].items():
        c, ls = styles[label]
        ax.plot(res.t / res.t[-1], res.emf, color=c, ls=ls, lw=1.6, label=label)
    eo = lenz["edge_on"]
    ax.plot(eo.t / eo.t[-1], eo.emf, color="#777", lw=1.2, ls=":", label="N approaches, coil edge-on (θ=90°)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title("Lenz's law: the sign of ε follows the sign of the flux change")
    ax.set_xlabel("Normalised time t / T")
    ax.set_ylabel("Induced EMF ε (V)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def scene_snapshot(res, index: int | None = None):
    """Side view at one instant: magnet, computed dipole field lines, coil and current."""
    if index is None:
        index = int(np.argmax(np.abs(res.emf)))
    g = SceneGeometry.from_result(res)
    X = res.x[index] * CM
    I = res.current[index]
    fig, ax = plt.subplots(figsize=(8.0, 8.0 / 2.1 + 0.6))
    fx, fy, sz = g.field_lines(X)
    ax.plot(fx, fy, color=COLORS["field"], lw=0.9)
    # arrowheads along the field lines
    for k in np.where(sz > 0)[0]:
        if k > 0 and np.isfinite(fx[k]) and np.isfinite(fx[k - 1]):
            ax.annotate("", xy=(fx[k], fy[k]), xytext=(fx[k - 1], fy[k - 1]),
                        arrowprops=dict(arrowstyle="-|>", color=COLORS["field"], lw=0.9))
    front, back, drawn = g.coil_turns(res.params.N)
    ax.plot(*back, color=COLORS["coil"], lw=1.2, alpha=0.35)
    ax.plot(*front, color=COLORS["coil"], lw=1.8)
    halves = g.magnet_halves(X)
    for pole in ("N", "S"):
        ax.fill(*halves[pole], color=COLORS[pole], ec="k", lw=0.8)
        ax.text(*halves[pole + "_label"], pole, color="white", ha="center", va="center", fontweight="bold")
    nx, ny = g.normal_arrow()
    ax.annotate("", xy=(nx[1], ny[1]), xytext=(nx[0], ny[0]),
                arrowprops=dict(arrowstyle="->", color="#555", ls="--"))
    ax.text(nx[1], ny[1], " n̂", color="#555", va="center")
    if abs(I) > 1e-3 * max(np.abs(res.current).max(), 1e-30):
        (tx, ty) = g.current_marker_points()
        top, bot = ("⊙", "⊗") if I > 0 else ("⊗", "⊙")
        ax.text(tx[0], ty[0], top, fontsize=16, ha="center", va="center", color=COLORS["current"])
        ax.text(tx[1], ty[1], bot, fontsize=16, ha="center", va="center", color=COLORS["current"])
    ax.set_xlim(*g.xlim)
    ax.set_ylim(*g.ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    sense = current_sense_from_left(I, res.params.theta_deg)
    ax.set_title(f"t = {res.t[index]:.4f} s, ε = {res.emf[index]:+.4g} V, I = {I*1e3:+.4g} mA "
                 f"(current {sense} seen from the left)", fontsize=9)
    fig.text(0.01, 0.01, f"Field lines: exact lines r = C sin²φ of the point-dipole model.  "
             f"Coil: {res.params.N} turns ({drawn} drawn).  ⊙ out of page, ⊗ into page.",
             fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def overview_figure(res):
    """Three stacked panels: flux, EMF and current vs time."""
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 7.2), sharex=True)
    axs[0].plot(res.t, res.flux * 1e6, color=COLORS["flux"])
    axs[0].set_ylabel("Φ (µWb)")
    axs[0].set_title(f"Flux, EMF and current — {MOTION_MODES[res.params.motion]}")
    axs[1].plot(res.t, res.emf, color=COLORS["emf"])
    axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_ylabel("ε (V)")
    axs[2].plot(res.t, res.current * 1e3, color=COLORS["current"])
    axs[2].axhline(0, color="k", lw=0.6)
    axs[2].set_ylabel("I (mA)")
    axs[2].set_xlabel("Time t (s)")
    fig.text(0.01, 0.005, _caption(res), fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def to_png_bytes(fig, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
