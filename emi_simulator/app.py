"""Electromagnetic Induction Virtual Lab — Streamlit application.

Run locally with:
    streamlit run app.py
then open http://localhost:8501 (Streamlit usually opens it for you).
Everything runs on your own computer; no internet connection is needed.
"""
from __future__ import annotations

import inspect
import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from emi import plots_mpl as pm
from emi import plots_plotly as pp
from emi.analysis import (SWEEP_INFO, coil_face_toward_magnet, fmt_si, lenz_experiment, observation_row,
                          params_table, sweep, sweep_conclusion)
from emi.explain import describe_change, describe_run
from emi.params import DEFAULTS, MOTION_MODES, POLARITIES, RANGES, R_REF, SimParams, validate
from emi.physics import dipole_axial_field, simulate, uniform_field_ramp
from emi.report import build_report
from emi.validation import run_all

st.set_page_config(page_title="EMI Virtual Lab — Faraday & Lenz", page_icon="🧲", layout="wide")

# ---------------------------------------------------------------------------
# Small compatibility helpers (Streamlit's API changed between versions)
# ---------------------------------------------------------------------------
def _stretch_kwargs(fn) -> dict:
    """Ask a Streamlit element to fill the container width on any version."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    w = params.get("width")
    if w is not None and isinstance(w.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


PLOT_KW = _stretch_kwargs(st.plotly_chart)
DF_KW = _stretch_kwargs(st.dataframe)


def show_plot(fig, key: str):
    st.plotly_chart(fig, theme=None, key=key, **PLOT_KW)


# ---------------------------------------------------------------------------
# Session state and defaults (UI units: cm for distances)
# ---------------------------------------------------------------------------
UI_DEFAULTS = {
    "B_ref": DEFAULTS.B_ref, "N": DEFAULTS.N, "v": DEFAULTS.v, "A": DEFAULTS.A,
    "theta_deg": DEFAULTS.theta_deg, "d_start_cm": DEFAULTS.d_start * 100, "d_min_cm": DEFAULTS.d_min * 100,
    "motion": DEFAULTS.motion, "polarity": DEFAULTS.polarity, "R": DEFAULTS.R,
    "manual_duration": False, "duration": 1.0,
}


def init_state():
    for k, v in UI_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("observations", [])
    st.session_state.setdefault("experiments", {})
    st.session_state.setdefault("lenz", None)
    st.session_state.setdefault("last_params", None)
    st.session_state.setdefault("last_metrics", None)
    st.session_state.setdefault("change_note", [])
    st.session_state.setdefault("png_zip", None)
    st.session_state.setdefault("report_html", None)


def reset_defaults():
    for k, v in UI_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["change_note"] = []


init_state()


# ---------------------------------------------------------------------------
# Cached computations
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=128)
def run_sim(pdict: dict):
    return simulate(SimParams(**pdict))


@st.cache_data(show_spinner=False, max_entries=48)
def run_sweep(pdict: dict, name: str, values: tuple):
    return sweep(SimParams(**pdict), name, list(values))


@st.cache_data(show_spinner=False, max_entries=16)
def lab_html(pdict: dict, theme: str, height: int):
    res = run_sim(pdict)
    fig, playback = pp.lab_animation(res, theme=theme, height=height)
    html = fig.to_html(include_plotlyjs=True, full_html=True, auto_play=True,
                       default_width="100%", default_height=f"{height}px",
                       config={"displaylogo": False, "responsive": True})
    return html, playback


@st.cache_data(show_spinner=False, max_entries=16)
def lab_figure(pdict: dict, theme: str, height: int):
    return pp.lab_animation(run_sim(pdict), theme=theme, height=height)


@st.cache_data(show_spinner=False)
def run_validation():
    return run_all()


# ---------------------------------------------------------------------------
# Sidebar: parameter controls
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.title("Parameter controls")
sb.caption("Change any value — the simulation, meters, graphs and explanation update immediately.")

sb.subheader("Magnet")
sb.slider("Magnet strength B₀ (T)", *RANGES["B_ref"], step=0.01, key="B_ref", format="%.2f",
          help=f"On-axis field of the magnet {R_REF*100:.0f} cm from its centre. It sets the dipole moment "
               "m = 2π r³ B₀ / μ₀. Changing B₀ scales the whole field; its shape stays the same.")
sb.selectbox("Pole facing the coil", options=list(POLARITIES), key="polarity", format_func=POLARITIES.get)

sb.subheader("Coil")
sb.slider("Number of turns N", 10, 1000, step=10, key="N")
sb.number_input("Coil area A (m²)", min_value=RANGES["A"][0], max_value=RANGES["A"][1], step=0.001,
                format="%.4f", key="A", help="Flat circular coil, radius a = √(A/π).")
sb.slider("Coil tilt θ (°)", 0.0, 180.0, step=1.0, key="theta_deg",
          help="Angle between the magnet's axis (direction of motion) and the coil's area vector n̂.")

sb.subheader("Motion")
sb.selectbox("Motion", options=list(MOTION_MODES), key="motion", format_func=MOTION_MODES.get)
sb.slider("Magnet speed v (m/s)", *RANGES["v"], step=0.01, key="v", format="%.2f")
sb.slider("Starting distance (cm)", RANGES["d_start"][0] * 100, RANGES["d_start"][1] * 100, step=1.0,
          key="d_start_cm", help="Distance from the magnet's centre to the coil's centre at the start.")
sb.slider("Closest distance (cm)", RANGES["d_min"][0] * 100, RANGES["d_min"][1] * 100, step=0.5,
          key="d_min_cm", disabled=st.session_state["motion"] == "pass",
          help="Where the magnet stops (approach), starts (move away) or turns back (push in, pull out). "
               "Not used for pass-through.")

sb.subheader("Circuit and timing")
sb.number_input("Total resistance R (Ω)", min_value=RANGES["R"][0], max_value=RANGES["R"][1],
                step=1.0, key="R", help="Coil + meter resistance. I = ε / R.")
sb.checkbox("Set simulation duration manually", key="manual_duration",
            help="Automatic duration covers the whole motion plus short rest periods.")
if st.session_state["manual_duration"]:
    sb.number_input("Duration (s)", min_value=RANGES["duration"][0], max_value=RANGES["duration"][1],
                    step=0.1, key="duration")

sb.button("Reset to defaults", on_click=reset_defaults, key="btn_reset", type="secondary")

sb.subheader("Display")
theme_label = sb.radio("Graph theme", ["Light", "Dark"], horizontal=True, key="plot_theme",
                       help="For the whole app, use the ⋮ menu → Settings → Theme.")
THEME = "dark" if theme_label == "Dark" else "light"
renderer = sb.selectbox("Animation player", ["Auto-play (recommended)", "Standard chart (press Play)"],
                        key="renderer",
                        help="If the animation does not appear, switch to the standard chart.")

# Build and validate parameters
raw = SimParams(
    B_ref=st.session_state["B_ref"], N=int(st.session_state["N"]), v=st.session_state["v"],
    A=st.session_state["A"], theta_deg=st.session_state["theta_deg"],
    d_start=st.session_state["d_start_cm"] / 100.0, d_min=st.session_state["d_min_cm"] / 100.0,
    motion=st.session_state["motion"], polarity=st.session_state["polarity"], R=st.session_state["R"],
    duration=st.session_state["duration"] if st.session_state["manual_duration"] else None,
)
params, input_msgs = validate(raw)
pdict = params.as_dict()
res = run_sim(pdict)
metrics = res.metrics()

# Track parameter changes for the explanation panel
if st.session_state["last_params"] != pdict:
    prev = SimParams(**st.session_state["last_params"]) if st.session_state["last_params"] else None
    note = describe_change(prev, st.session_state["last_metrics"], res)
    if note:
        st.session_state["change_note"] = note
    st.session_state["last_params"] = pdict
    st.session_state["last_metrics"] = metrics

for msg in input_msgs:
    sb.warning(msg)


def sweep_base() -> dict:
    """Current parameters with automatic duration (full trajectory for every swept value)."""
    return params.with_(duration=None).as_dict()


# ---------------------------------------------------------------------------
# Header and tabs
# ---------------------------------------------------------------------------
st.title("Electromagnetic induction virtual lab")
st.caption("Faraday's law and Lenz's law with a moving bar magnet and a coil — B.Tech Engineering Physics")

tabs = st.tabs(["Project overview", "Theory and equations", "Interactive simulation", "Parameter controls",
                "Live graphs", "Virtual experiments", "Observation table", "Results and conclusion",
                "References"])

# ---------------------------------------------------------------- 1. Overview
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            """
**What this lab does.** A bar magnet moves along the axis of a coil. As it moves, the magnetic
flux through the coil changes and an EMF is induced. You control the magnet, the coil and the
motion, and the simulator calculates the flux, EMF and current from first principles.

**How to use it**

1. Set the magnet, coil and motion in the sidebar.
2. Watch the animation in **Interactive simulation** — the voltmeter, galvanometer and the
   current arrows show what the coil is doing at each instant.
3. Study the graphs in **Live graphs**.
4. Run the four guided experiments in **Virtual experiments**; results go into the
   **Observation table** automatically.
5. Download the CSV and the report from **Results and conclusion**.

**What you should be able to show at the end**

* ε ∝ B₀, ε ∝ N and ε ∝ v (for the same magnet path).
* No change of flux → no EMF, even in a strong field or with a moving magnet.
* The induced current always opposes the *change* of flux (Lenz's law).
            """)
    with c2:
        st.markdown("**Current run at a glance**")
        st.metric("Peak |EMF|", fmt_si(metrics["max_abs_emf_V"], "V"))
        st.metric("Peak |current|", fmt_si(metrics["max_abs_current_A"], "A"))
        st.metric("Maximum flux per turn", fmt_si(metrics["max_abs_flux_Wb"], "Wb"))
        st.info("**Sign convention.** Positive flux points along the coil's area vector n̂. Positive EMF "
                "and current circulate around n̂ by the right-hand rule. With θ = 0, n̂ points away "
                "from the magnet's starting side, so positive current looks **clockwise from the "
                "left** (from the magnet).")

# ---------------------------------------------------------------- 2. Theory
with tabs[1]:
    st.subheader("Faraday's law")
    st.markdown("A changing magnetic flux through a circuit induces an EMF equal to the rate of change "
                "of flux linkage. For a coil of N turns:")
    st.latex(r"\mathcal{E} = -N\,\frac{d\Phi_B}{dt}")
    st.markdown("The magnetic flux through one turn is the field 'flowing' through its area:")
    st.latex(r"\Phi_B = \int \vec B\cdot d\vec A \quad\xrightarrow{\text{uniform } B}\quad \Phi_B = BA\cos\theta")
    st.markdown("For a uniform field that changes with time at fixed orientation, the EMF is")
    st.latex(r"\mathcal{E} = -NA\cos\theta\,\frac{dB}{dt}")
    st.subheader("Lenz's law (the minus sign)")
    st.markdown("The induced current flows in the direction that **opposes the change in flux** that "
                "produced it. If the flux through the coil increases, the induced current makes its own "
                "field against the increase; if the flux decreases, it makes a field that tries to keep "
                "it. An approaching N pole therefore meets an N face (repulsion), and a receding N pole "
                "is followed by an S face (attraction). Note that it is the *flux change* that is "
                "opposed: an edge-on coil sees no flux change and no current, even though the magnet moves.")

    st.subheader("The moving-magnet model used here")
    st.markdown(f"""
* **Magnet:** an ideal point magnetic dipole of moment *m* at the magnet's centre, on the x-axis.
  The control **B₀** is the field on the magnet's axis {R_REF*100:.0f} cm from its centre, so""")
    st.latex(r"m = \frac{2\pi r_\text{ref}^3 B_0}{\mu_0}, \qquad B_\text{axis}(r) = \frac{\mu_0 m}{2\pi r^3}")
    st.markdown("""
* **Coil:** N thin circular turns of radius a = √(A/π) at the origin, normal n̂ at angle θ to the
  x-axis. All turns link the same flux.
* **Flux:** computed exactly for any position and tilt using the reciprocity theorem
  (Φ = **m**·**B**_loop / I) with the closed-form field of a circular loop (complete elliptic
  integrals K and E). On the axis (θ = 0) this reduces to""")
    st.latex(r"\Phi_B(x) = \frac{\mu_0\, m\, a^2}{2\,(a^2 + x^2)^{3/2}}")
    st.markdown("* **Rate of change:** Φ is sampled in time and differentiated with second-order central "
                "differences:")
    st.latex(r"\frac{d\Phi}{dt}\Big|_i \approx \frac{\Phi_{i+1}-\Phi_{i-1}}{t_{i+1}-t_{i-1}}, \qquad "
             r"\mathcal{E}_i = -N\frac{d\Phi}{dt}\Big|_i, \qquad I_i = \frac{\mathcal{E}_i}{R}")
    st.markdown("Because the magnet moves at speed v along a fixed path, the chain rule gives "
                "dΦ/dt = (dΦ/dx)·v — the reason the EMF is proportional to speed.")
    st.subheader("Assumptions and limitations")
    st.markdown("""
* Point dipole: accurate when the magnet is small compared with its distance from the coil; only
  approximate when the magnet is inside or very close to the coil. The drawn magnet size is illustrative.
* Thin coil (all turns link the same flux); a long solenoid would need the flux summed turn by turn.
* Resistive circuit only: coil self-inductance and the force on the magnet are ignored, so the
  magnet's speed stays constant and I = ε/R exactly.
* Constant-speed motion with instantaneous start/stop and turnaround.
* Peak EMF is the largest instantaneous |ε|; average EMF is −NΔΦ/Δt over the whole motion.
""")

# ---------------------------------------------------------------- 3. Simulation
with tabs[2]:
    for msg in res.messages:
        st.warning(msg)
    height = 760
    if renderer.startswith("Auto"):
        try:
            html, playback = lab_html(pdict, THEME, height)
            components.html(html, height=height + 20, scrolling=False)
        except Exception as exc:  # pragma: no cover - depends on the browser/Streamlit build
            st.error(f"The auto-play player could not be shown ({exc}). Using the standard chart.")
            fig, playback = lab_figure(pdict, THEME, height)
            show_plot(fig, "lab_fallback")
    else:
        fig, playback = lab_figure(pdict, THEME, height)
        show_plot(fig, "lab_native")
    st.caption(f"Playback: {playback:.1f} s on screen for {res.duration:.4g} s of simulated time "
               f"(×{playback/res.duration:.3g} slow motion). Field lines are calculated from the dipole "
               "model. The flux bar inside the coil darkens as |Φ| grows (purple = along n̂, blue = against).")

    cols = st.columns(6)
    cols[0].metric("Peak +EMF", fmt_si(metrics["peak_pos_emf_V"], "V"))
    cols[1].metric("Peak −EMF", fmt_si(metrics["peak_neg_emf_V"], "V"))
    cols[2].metric("Average EMF", fmt_si(metrics["average_emf_V"], "V"))
    cols[3].metric("Peak |I|", fmt_si(metrics["max_abs_current_A"], "A"))
    cols[4].metric("Max flux / turn", fmt_si(metrics["max_abs_flux_Wb"], "Wb"))
    cols[5].metric("ΔΦ (end − start)", fmt_si(metrics["delta_flux_Wb"], "Wb"))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### What is happening")
        for line in describe_run(res):
            st.markdown(line)
        if st.session_state["change_note"]:
            st.info("\n\n".join(st.session_state["change_note"]))
    with right:
        st.markdown("#### Inspect one instant")
        t_pick = st.slider("Time (s)", 0.0, float(res.duration), float(metrics["time_of_max_abs_emf_s"]),
                           step=float(res.duration) / 500, format="%.4f", key=f"t_pick_{hash(tuple(pdict.items()))}")
        s = res.sample_at(t_pick)
        face = coil_face_toward_magnet(s["current"], s["x"], params.theta_deg)
        st.markdown(
            f"Magnet centre at **x = {s['x']*100:+.2f} cm**  \n"
            f"Flux per turn **Φ = {fmt_si(s['flux'], 'Wb')}**  \n"
            f"EMF **ε = {fmt_si(s['emf'], 'V')}**  \n"
            f"Current **I = {fmt_si(s['current'], 'A')}**  \n"
            f"Coil face toward magnet: **{face}**")
        if st.button("Add this run to the observation table", key="btn_add_sim"):
            st.session_state["observations"].append(observation_row("Free run", params, metrics))
            st.success("Added. See the Observation table section.")

# ---------------------------------------------------------------- 4. Parameter controls
with tabs[3]:
    st.markdown("All controls are in the sidebar on the left (use **›** at the top-left if it is hidden). "
                "Values are checked and clamped to safe ranges; any correction is shown in the sidebar.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Current settings")
        st.table(pd.DataFrame(params_table(params), columns=["Parameter", "Value"]).set_index("Parameter"))
    with c2:
        st.markdown("#### Derived quantities")
        d_close = params.d_min if params.motion != "pass" else None
        derived = [
            ("Dipole moment m", f"{metrics['dipole_moment_Am2']:.4g} A·m²"),
            ("Coil radius a", f"{res.a*100:.2f} cm"),
            ("Field at coil centre, magnet at start", fmt_si(float(dipole_axial_field(abs(res.m), params.d_start)), "T")),
            ("Field at coil centre, magnet at closest point",
             fmt_si(float(dipole_axial_field(abs(res.m), d_close)), "T") if d_close else
             "magnet passes through the centre"),
            ("Mean normal field over the coil at maximum flux (Φmax/A)",
             fmt_si(metrics["mean_normal_field_at_max_flux_T"], "T")),
            ("Motion time", f"{metrics['motion_time_s']:.4g} s"),
            ("Simulated duration", f"{res.duration:.4g} s, {len(res.t)} samples (Δt = {res.t[1]-res.t[0]:.3g} s)"),
        ]
        st.table(pd.DataFrame(derived, columns=["Quantity", "Value"]).set_index("Quantity"))
    st.markdown("#### Three different 'field strengths' — don't mix them up")
    st.markdown(f"""
* **Magnet strength B₀** (the control): a property of the magnet — its on-axis field {R_REF*100:.0f} cm from
  its centre. Changing it scales the dipole moment m and the field everywhere by the same factor.
* **Field at the coil**: depends on B₀ **and** on where the magnet is; it changes during the motion
  (falls off roughly as 1/r³).
* **Field distribution**: the shape of the field (the field-line pattern). It is fixed by the dipole
  model and does not change with B₀.

When you vary B₀, the coil (N, A, θ), the speed, the path and R are held constant.""")

# ---------------------------------------------------------------- 5. Live graphs
B_GRID = tuple(np.round(np.linspace(0.1, 1.0, 10), 3))
N_GRID = tuple(range(100, 1001, 100))
V_GRID = tuple(np.round(np.linspace(0.5, 5.0, 10), 3))

with tabs[4]:
    st.caption("All curves are computed by the model. Hover for values; use the camera icon on a graph to "
               "save it as PNG, or download every graph below.")
    base = sweep_base()
    sw_B = run_sweep(base, "B_ref", B_GRID)
    sw_N = run_sweep(base, "N", N_GRID)
    sw_v = run_sweep(base, "v", V_GRID)
    g1, g2 = st.columns(2)
    with g1:
        show_plot(pp.emf_vs_time(res, THEME), "g_emf")
    with g2:
        show_plot(pp.sweep_figure(sw_B, params.B_ref, THEME, "2"), "g_B")
    g3, g4 = st.columns(2)
    with g3:
        show_plot(pp.sweep_figure(sw_N, params.N, THEME, "3"), "g_N")
    with g4:
        show_plot(pp.sweep_figure(sw_v, params.v, THEME, "4"), "g_v")
    g5, g6 = st.columns(2)
    with g5:
        show_plot(pp.flux_vs_time(res, THEME), "g_flux")
    with g6:
        show_plot(pp.current_vs_time(res, THEME), "g_I")

    if st.button("Prepare all graphs as PNG (ZIP)", key="btn_png"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("1_emf_vs_time.png", pm.to_png_bytes(pm.emf_vs_time(res)))
            zf.writestr("2_peak_emf_vs_B.png", pm.to_png_bytes(pm.sweep_plot(sw_B, params.B_ref)))
            zf.writestr("3_peak_emf_vs_N.png", pm.to_png_bytes(pm.sweep_plot(sw_N, params.N)))
            zf.writestr("4_peak_emf_vs_v.png", pm.to_png_bytes(pm.sweep_plot(sw_v, params.v)))
            zf.writestr("5_flux_vs_time.png", pm.to_png_bytes(pm.flux_vs_time(res)))
            zf.writestr("6_current_vs_time.png", pm.to_png_bytes(pm.current_vs_time(res)))
            zf.writestr("7_side_view_at_peak.png", pm.to_png_bytes(pm.scene_snapshot(res)))
        st.session_state["png_zip"] = buf.getvalue()
    if st.session_state["png_zip"]:
        st.download_button("Download graphs (ZIP)", st.session_state["png_zip"], file_name="emi_graphs.zip",
                           mime="application/zip", key="dl_png")

# ---------------------------------------------------------------- 6. Experiments
def parse_values(text: str, name: str) -> tuple[list[float], list[str]]:
    lo, hi = RANGES[name]
    vals, errs = [], []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            x = float(tok)
        except ValueError:
            errs.append(f"'{tok}' is not a number.")
            continue
        if not (lo <= x <= hi):
            errs.append(f"{x:g} is outside the allowed range {lo:g} to {hi:g}.")
            continue
        vals.append(int(round(x)) if name == "N" else x)
    vals = sorted(set(vals))
    if len(vals) < 2:
        errs.append("Enter at least two different valid values.")
    return vals, errs


def sweep_experiment(code: str, name: str, default_text: str, intro: str):
    info = SWEEP_INFO[name]
    st.markdown(intro)
    st.markdown(f"**Held constant:** {info['held']}. **Current fixed values:** B₀ = {params.B_ref:g} T, "
                f"N = {params.N}, v = {params.v:g} m/s, A = {params.A:g} m², θ = {params.theta_deg:g}°, "
                f"motion = {MOTION_MODES[params.motion].lower()}, {params.polarity} pole facing, R = {params.R:g} Ω.")
    text = st.text_input(f"Values of {info['label']} ({info['unit']}), separated by commas",
                         default_text, key=f"vals_{code}")
    vals, errs = parse_values(text, name)
    for e in errs:
        st.error(e)
    if st.button(f"Run experiment {code}", key=f"run_{code}", type="primary", disabled=bool(errs)):
        sw = sweep(params.with_(duration=None), name, vals)
        st.session_state["experiments"][code] = sw
        for row in sw["rows"]:
            st.session_state["observations"].append(
                observation_row(f"Experiment {code}", row["params"], row, info["label"], row["value"]))
        st.success(f"Experiment {code} recorded ({len(vals)} runs added to the observation table).")
    sw = st.session_state["experiments"].get(code)
    if sw is None or sw["name"] != name:
        st.info("Press the button to run the experiment.")
        return
    c1, c2 = st.columns([3, 2])
    with c1:
        show_plot(pp.sweep_figure(sw, None, THEME), f"exp_{code}")
    with c2:
        obs, theory = sweep_conclusion(sw)
        st.markdown(f"**Observation.** {obs}")
        st.markdown(f"**Comparison with theory.** {theory}")
    table = pd.DataFrame({
        info["label"] + f" ({info['unit']})": sw["x"],
        "Peak +EMF (V)": sw["peak_pos"], "Peak −EMF (V)": sw["peak_neg"], "Max |EMF| (V)": sw["peak"],
        "Max |I| (A)": sw["max_I"],
        "ΔΦ per turn (Wb)": [r["delta_flux_Wb"] for r in sw["rows"]],
        f"|EMF| / {info['symbol']}": sw["peak"] / sw["x"],
    })
    st.dataframe(table, hide_index=True, **DF_KW)
    st.download_button(f"Download experiment {code} (CSV)", table.to_csv(index=False).encode(),
                       file_name=f"experiment_{code}.csv", mime="text/csv", key=f"dl_{code}")


with tabs[5]:
    if params.motion == "stationary":
        st.warning("The motion is set to 'stationary', so every experiment will give zero EMF. Choose a "
                   "moving mode in the sidebar for experiments A–C.")
    eA, eB, eC, eD = st.tabs(["A · Magnet strength", "B · Number of turns", "C · Magnet speed", "D · Lenz's law"])
    with eA:
        sweep_experiment("A", "B_ref", "0.1, 0.2, 0.4, 0.6, 0.8, 1.0",
                         "**Aim:** find how the peak EMF depends on the strength of the magnet. "
                         "**Why it should grow:** a stronger magnet has a proportionally stronger field "
                         "everywhere, so the same motion changes the flux by a proportionally larger amount "
                         "in the same time.")
    with eB:
        sweep_experiment("B", "N", "50, 100, 200, 400, 600, 800, 1000",
                         "**Aim:** show that the EMF is proportional to the number of turns. The flux per "
                         "turn and the motion are identical in every run, so only N changes.")
    with eC:
        sweep_experiment("C", "v", "0.25, 0.5, 1, 2, 3, 4, 5",
                         "**Aim:** find how the peak EMF depends on the magnet's speed. The magnet follows "
                         "the same path every time, so the flux at each position is the same; a faster "
                         "magnet simply goes through those positions sooner, making dΦ/dt larger.")
    with eD:
        st.markdown("**Aim:** verify Lenz's law. The magnet approaches or moves away with either pole facing "
                    "the coil (θ = 0, other settings from the sidebar). Two extra runs show that the "
                    "induced current opposes the change of *flux*, not motion as such.")
        if st.button("Run experiment D", key="run_D", type="primary"):
            lz = lenz_experiment(params)
            st.session_state["lenz"] = lz
            for label, r in lz["results"].items():
                st.session_state["observations"].append(
                    observation_row("Experiment D", r.params, r.metrics(), "case", label))
            st.success("Experiment D recorded (4 runs added to the observation table).")
        lz = st.session_state["lenz"]
        if lz is None:
            st.info("Press the button to run the experiment.")
        else:
            show_plot(pp.lenz_figure(lz, THEME), "exp_D")
            st.dataframe(pd.DataFrame(lz["cases"]), hide_index=True, **DF_KW)
            ramp = lz["ramp"]
            eo = lz["edge_on"]
            st.markdown(f"""
**Observations**

* Approaching N pole: the flux along n̂ increases, ε is negative, and the coil face toward the
  magnet becomes **N** — it repels the magnet. Moving the N pole away reverses everything: the face
  becomes **S** and attracts the magnet.
* Reversing the magnet (S pole facing) reverses the EMF for the *same* motion. The current direction
  follows the change of flux, not the direction of motion.
* **Flux, not motion:** with the coil edge-on (θ = 90°) the magnet moves exactly as before, but the
  flux stays at {fmt_si(float(np.abs(eo.flux).max()), 'Wb')}, so ε = {fmt_si(float(np.abs(eo.emf).max()), 'V')}:
  no current and no opposing force.
* **Change without motion:** nothing moves, but a uniform field rising at 0.5 T/s through the coil
  (N = {params.N}, A = {params.A:g} m², θ = 0) gives a steady ε = {fmt_si(float(np.mean(ramp['emf'])), 'V')}
  (theory −NA dB/dt = {fmt_si(float(ramp['emf_theory']), 'V')}). The current opposes the growing field
  even though there is no motion to oppose.

**Conclusion.** The induced current always flows so that its magnetic field opposes the change in flux
through the coil — Lenz's law, which is the minus sign in ε = −N dΦ/dt.""")

# ---------------------------------------------------------------- 7. Observation table
with tabs[6]:
    obs = st.session_state["observations"]
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("Add current run", key="btn_add_obs"):
        obs.append(observation_row("Free run", params, metrics))
    if c2.button("Clear table", key="btn_clear_obs"):
        st.session_state["observations"] = []
        obs = st.session_state["observations"]
    if not obs:
        st.info("The table is empty. Run an experiment or add the current run.")
    else:
        df = pd.DataFrame(obs)
        df.insert(0, "S.No.", range(1, len(df) + 1))
        st.dataframe(df, hide_index=True, **DF_KW)
        st.download_button("Download observation table (CSV)", df.to_csv(index=False).encode(),
                           file_name="observation_table.csv", mime="text/csv", key="dl_obs")

# ---------------------------------------------------------------- 8. Results & conclusion
with tabs[7]:
    st.markdown("#### Results of the current run")
    rows = [("Peak positive EMF", fmt_si(metrics["peak_pos_emf_V"], "V")),
            ("Peak negative EMF", fmt_si(metrics["peak_neg_emf_V"], "V")),
            ("Maximum |EMF|", fmt_si(metrics["max_abs_emf_V"], "V")),
            ("Average EMF over the motion, −NΔΦ/Δt", fmt_si(metrics["average_emf_V"], "V")),
            ("Mean |EMF| while moving", fmt_si(metrics["mean_abs_emf_V"], "V")),
            ("Maximum |current|", fmt_si(metrics["max_abs_current_A"], "A")),
            ("Flux per turn: start → end", f"{fmt_si(metrics['flux_initial_Wb'], 'Wb')} → {fmt_si(metrics['flux_final_Wb'], 'Wb')}"),
            ("Change in flux per turn ΔΦ", fmt_si(metrics["delta_flux_Wb"], "Wb")),
            ("Net charge ∫I dt (= −NΔΦ/R)", fmt_si(metrics["net_charge_C"], "C"))]
    st.table(pd.DataFrame(rows, columns=["Quantity", "Value"]).set_index("Quantity"))
    ts = pd.DataFrame({"t (s)": res.t, "x (m)": res.x, "flux per turn (Wb)": res.flux,
                       "dPhi/dt (Wb/s)": res.dflux_dt, "EMF (V)": res.emf, "current (A)": res.current})
    st.download_button("Download time series of this run (CSV)", ts.to_csv(index=False).encode(),
                       file_name="emi_time_series.csv", mime="text/csv", key="dl_ts")

    st.markdown("#### Model validation against theory")
    st.caption("These checks run the simulator on cases with known answers.")
    val = run_validation()
    st.dataframe(pd.DataFrame(val), hide_index=True, **DF_KW)
    n_ok = sum(c["Passed"] for c in val)
    if n_ok == len(val):
        st.success(f"All {n_ok} checks passed.")
    else:
        st.error(f"{n_ok} of {len(val)} checks passed.")

    st.markdown("#### Experiment report")
    name = st.text_input("Name / roll number for the report (optional)", key="report_name")
    if st.button("Build report", key="btn_report", type="primary"):
        with st.spinner("Building report…"):
            sweeps = {sw["name"]: sw for sw in st.session_state["experiments"].values()}
            st.session_state["report_html"] = build_report(
                res, student=name, sweeps=sweeps or None, lenz=st.session_state["lenz"],
                observations=st.session_state["observations"] or None, validation=val)
    if st.session_state["report_html"]:
        st.download_button("Download report (HTML)", st.session_state["report_html"].encode("utf-8"),
                           file_name="emi_experiment_report.html", mime="text/html", key="dl_report")
        st.caption("Open the file in a browser; use Print → Save as PDF if you need a PDF.")

    st.markdown("#### Conclusion")
    st.markdown("""
1. The EMF is induced only while the flux through the coil is changing; a stationary magnet gives
   zero EMF however strong it is.
2. For the same magnet path, the peak EMF is directly proportional to the magnet strength B₀, the
   number of turns N and the speed v (experiments A, B, C).
3. The EMF changes sign when the flux stops increasing and starts decreasing, and when the motion or
   the magnet's polarity is reversed.
4. The induced current always opposes the change of flux that causes it (Lenz's law, experiment D).
5. The total charge through the circuit, −NΔΦ/R, depends only on the flux change, not on how fast it
   happens; speed affects the peak EMF, not the charge.""")

# ---------------------------------------------------------------- 9. References
with tabs[8]:
    st.markdown("""
1. D. J. Griffiths, *Introduction to Electrodynamics*, 4th ed., Cambridge University Press — ch. 5
   (field of a current loop, magnetic dipole) and ch. 7 (Faraday's law, Lenz's law).
2. D. Halliday, R. Resnick and J. Walker, *Fundamentals of Physics*, Wiley — chapter on induction and inductance.
3. H. C. Verma, *Concepts of Physics*, Vol. 2, Bharati Bhawan — chapter on electromagnetic induction.
4. J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley — §5.5 (field of a circular loop with
   elliptic integrals), §5.6 (dipole field).
5. J. C. Simpson, J. E. Lane, C. D. Immer and R. C. Youngquist, "Simple analytic expressions for the
   magnetic field of a circular current loop", NASA Technical Report NASA/TM-2013-217919 (2001).
6. M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions* — complete elliptic integrals K and E.
7. NumPy `numpy.gradient` documentation — second-order central differences.
8. SciPy `scipy.special.ellipk` / `ellipe` documentation.
""")
