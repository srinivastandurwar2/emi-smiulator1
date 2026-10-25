"""Experiments built on top of the physics engine.

Every number produced here comes from :func:`emi.physics.simulate`; nothing is
fabricated. Parameter sweeps always use the automatic duration so that the full
trajectory is simulated for every value.
"""
from __future__ import annotations

import numpy as np

from .params import MOTION_MODES, POLARITIES, SimParams
from .physics import simulate, uniform_field_ramp

SWEEP_INFO = {
    "B_ref": {"label": "Magnet field strength B₀", "unit": "T", "symbol": "B₀",
              "held": "coil turns N, coil area A, tilt θ, speed v, trajectory, resistance R"},
    "N": {"label": "Number of turns N", "unit": "turns", "symbol": "N",
          "held": "magnet strength B₀ (so flux per turn), coil area A, tilt θ, speed v, trajectory, R"},
    "v": {"label": "Magnet speed v", "unit": "m/s", "symbol": "v",
          "held": "magnet strength B₀ (field distribution), coil turns N, area A, tilt θ, trajectory, R"},
    "A": {"label": "Coil area A", "unit": "m²", "symbol": "A",
          "held": "magnet strength B₀, N, θ, v, trajectory, R"},
    "theta_deg": {"label": "Coil tilt θ", "unit": "°", "symbol": "θ",
                  "held": "magnet strength B₀, N, A, v, trajectory, R"},
}


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------
def sweep(base: SimParams, name: str, values) -> dict:
    """Run one simulation per value of parameter ``name`` and collect metrics."""
    values = [float(v) for v in values]
    rows = []
    for val in values:
        kw = {name: int(round(val)) if name == "N" else val, "duration": None}
        res = simulate(base.with_(**kw))
        m = res.metrics()
        rows.append({"value": float(getattr(res.params, name)), "params": res.params, **m})
    x = np.array([r["value"] for r in rows])
    y = np.array([r["max_abs_emf_V"] for r in rows])
    fit = proportional_fit(x, y)
    return {"name": name, "x": x, "peak": y,
            "peak_pos": np.array([r["peak_pos_emf_V"] for r in rows]),
            "peak_neg": np.array([r["peak_neg_emf_V"] for r in rows]),
            "max_I": np.array([r["max_abs_current_A"] for r in rows]),
            "rows": rows, "fit": fit}


def proportional_fit(x, y) -> dict:
    """Least-squares fits y = k x (through origin) and y = k x + c.

    Returns slope, R² of the proportional model, the general intercept and the
    spread of the ratio y/x (0 % means perfectly proportional).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out = {"k": float("nan"), "r2": float("nan"), "slope": float("nan"),
           "intercept": float("nan"), "ratio_spread_pct": float("nan")}
    if len(x) < 2 or np.allclose(x, x[0]):
        return out
    k = float(np.dot(x, y) / np.dot(x, x))
    ss_res = float(np.sum((y - k * x) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    slope, intercept = np.polyfit(x, y, 1)
    nz = x != 0
    ratios = y[nz] / x[nz]
    spread = float(100 * (ratios.max() - ratios.min()) / abs(ratios.mean())) if ratios.size and ratios.mean() != 0 else float("nan")
    out.update(k=k, r2=r2, slope=float(slope), intercept=float(intercept), ratio_spread_pct=spread)
    return out


def sweep_conclusion(result: dict) -> tuple[str, str]:
    """(observation, theory comparison) sentences for a sweep."""
    name = result["name"]
    info = SWEEP_INFO.get(name, {"label": name, "unit": "", "symbol": name})
    x, y, fit = result["x"], result["peak"], result["fit"]
    if len(x) < 2:
        return ("Run at least two values to see a trend.", "")
    if np.all(y == 0):
        return ("The peak EMF is zero for every value.",
                "With no change of flux (stationary magnet or coil edge-on) Faraday's law gives ε = 0 "
                "regardless of this parameter.")
    i0, i1 = int(np.argmin(x)), int(np.argmax(x))
    gain = y[i1] / y[i0] if y[i0] > 0 else float("inf")
    xr = x[i1] / x[i0] if x[i0] > 0 else float("inf")
    obs = (f"When {info['symbol']} increased from {x[i0]:g} to {x[i1]:g} {info['unit']} "
           f"(×{xr:.2f}), the peak EMF changed from {fmt_si(y[i0], 'V')} to {fmt_si(y[i1], 'V')} "
           f"(×{gain:.2f}).")
    if name in ("B_ref", "N", "v"):
        why = {
            "B_ref": "A stronger magnet scales the field everywhere by the same factor, so the flux "
                     "Φ(x) and its rate of change dΦ/dt scale with B₀.",
            "N": "Each turn links the same flux, so the total flux linkage NΦ — and therefore "
                 "ε = −N dΦ/dt — is proportional to N.",
            "v": "On the same path, the flux depends only on position: dΦ/dt = (dΦ/dx)·v. "
                 "Doubling v halves the time for the same flux change, doubling the EMF.",
        }[name]
        theory = (f"Theory predicts peak EMF ∝ {info['symbol']}. Fit through the origin: "
                  f"ε_peak = {fit['k']:.4g}·{info['symbol']} with R² = {fit['r2']:.6f}; the ratio "
                  f"ε_peak/{info['symbol']} varies by only {fit['ratio_spread_pct']:.3f} %. "
                  f"Result: consistent with direct proportionality. {why}")
    elif name == "A":
        theory = ("Theory: a larger coil collects more flux, but the dipole field is not uniform, "
                  "so the peak EMF is not exactly proportional to A (it would be in a uniform field).")
    else:
        theory = ("Theory: only the field component along the coil's normal contributes to flux. "
                  "Near the axis this behaves roughly like cos θ, reaching zero at θ = 90°.")
    return obs, theory


# ---------------------------------------------------------------------------
# Lenz's law experiment
# ---------------------------------------------------------------------------
def current_sense_from_left(I: float, theta_deg: float) -> str:
    """Apparent circulation of current I when the coil is viewed from the left (−x side).

    Positive current is right-handed about n̂. Looking from the left means
    looking along +x; if n̂ has a positive x-component, positive current looks
    clockwise.
    """
    c = np.cos(np.deg2rad(theta_deg))
    if abs(I) < 1e-15 or abs(c) < 1e-9:
        return "none"
    return "clockwise" if I * c > 0 else "anticlockwise"


def coil_face_toward_magnet(I: float, x_mag: float, theta_deg: float) -> str:
    """Which induced pole ('N'/'S') the coil presents to the magnet, or '–'."""
    c = np.cos(np.deg2rad(theta_deg))
    side = np.sign(x_mag * c)  # magnet's side of the coil plane, measured along n̂
    if abs(I) < 1e-15 or side == 0:
        return "–"
    return "N" if np.sign(I) == side else "S"


def dominant_index(emf: np.ndarray) -> int:
    return int(np.argmax(np.abs(emf)))


LENZ_CASES = [
    ("N", "approach", "N pole approaches"),
    ("N", "recede", "N pole moves away"),
    ("S", "approach", "S pole approaches"),
    ("S", "recede", "S pole moves away"),
]


def lenz_experiment(base: SimParams) -> dict:
    """Run the four classic Lenz cases plus two 'flux, not motion' demonstrations."""
    base = base.with_(theta_deg=0.0, duration=None)
    cases = []
    results = {}
    for pol, mode, label in LENZ_CASES:
        res = simulate(base.with_(polarity=pol, motion=mode))
        i = dominant_index(res.emf)
        I = float(res.current[i])
        flux_trend = "increasing" if res.flux[-1] > res.flux[0] else "decreasing"
        face = coil_face_toward_magnet(I, float(res.x[i]), 0.0)
        # Like poles facing each other repel, unlike poles attract.
        if face == "–":
            effect = "no interaction"
        else:
            effect = ("repels" if face == pol else "attracts") + \
                     (", opposing the approach" if mode == "approach" else ", opposing the separation")
        cases.append({
            "Case": label,
            "Facing pole": pol,
            "Motion": mode,
            "Flux along n̂": flux_trend,
            "Dominant EMF (V)": float(res.emf[i]),
            "Current (A)": I,
            "Current seen from magnet side": current_sense_from_left(I, 0.0),
            "Coil face toward magnet": face,
            "Effect on magnet": effect,
        })
        results[label] = res

    # Demonstration 1: magnet moves, but the coil is edge-on (θ = 90°) -> no flux change, no current
    edge_on = simulate(base.with_(theta_deg=90.0, motion="approach", polarity="N"))
    # Demonstration 2: no motion at all, but a changing uniform field -> current still induced
    ramp = uniform_field_ramp(N=base.N, A=base.A, theta_deg=0.0, B0=0.0, dBdt=0.5, R=base.R)
    return {"cases": cases, "results": results, "edge_on": edge_on, "ramp": ramp}


# ---------------------------------------------------------------------------
# Observation table rows
# ---------------------------------------------------------------------------
def observation_row(experiment: str, p: SimParams, m: dict, variable: str = "", value=None) -> dict:
    """One row of the observation table from parameters and a metrics dict."""
    return {
        "Experiment": experiment,
        "Varied": variable,
        "Value": value if value is not None else "",
        "B0 (T)": p.B_ref,
        "N (turns)": p.N,
        "v (m/s)": p.v,
        "A (m^2)": p.A,
        "theta (deg)": p.theta_deg,
        "Start distance (m)": p.d_start,
        "Closest distance (m)": p.d_min if p.motion != "pass" else 0.0,
        "Motion": p.motion,
        "Facing pole": p.polarity,
        "Duration (s)": "auto" if p.duration is None else p.duration,
        "R (ohm)": p.R,
        "Peak +EMF (V)": m["peak_pos_emf_V"],
        "Peak -EMF (V)": m["peak_neg_emf_V"],
        "Max |EMF| (V)": m["max_abs_emf_V"],
        "Average EMF (V)": m["average_emf_V"],
        "Max |I| (A)": m["max_abs_current_A"],
        "Delta flux per turn (Wb)": m["delta_flux_Wb"],
        "Max |flux| per turn (Wb)": m["max_abs_flux_Wb"],
        "Net charge (C)": m["net_charge_C"],
    }


def params_table(p: SimParams) -> list[tuple[str, str]]:
    """Readable (name, value) pairs for reports and the UI."""
    return [
        ("Magnet strength B₀ (on-axis field 2 cm from magnet centre)", f"{p.B_ref:g} T"),
        ("Number of turns N", f"{p.N}"),
        ("Magnet speed v", f"{p.v:g} m/s"),
        ("Coil area A", f"{p.A:g} m²  (radius {np.sqrt(p.A/np.pi)*100:.2f} cm)"),
        ("Coil tilt θ (magnet axis vs coil normal)", f"{p.theta_deg:g}°"),
        ("Starting distance", f"{p.d_start*100:.1f} cm"),
        ("Closest distance", "passes through centre" if p.motion == "pass" else f"{p.d_min*100:.1f} cm"),
        ("Motion", MOTION_MODES[p.motion]),
        ("Magnet polarity", POLARITIES[p.polarity]),
        ("Circuit resistance R", f"{p.R:g} Ω"),
        ("Simulated duration", "automatic" if p.duration is None else f"{p.duration:g} s"),
    ]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
_PREFIXES = [(1e0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")]


def fmt_si(value: float, unit: str, digits: int = 4) -> str:
    """Format with an SI prefix, e.g. 0.00123 V -> '1.23 mV'."""
    if value is None or not np.isfinite(value):
        return "–"
    v = float(value)
    if abs(v) < 1e-12:
        return f"0 {unit}"
    for scale, prefix in _PREFIXES:
        if abs(v) >= scale or scale == 1e-9:
            return f"{v/scale:.{digits}g} {prefix}{unit}"
    return f"{v:.{digits}g} {unit}"
