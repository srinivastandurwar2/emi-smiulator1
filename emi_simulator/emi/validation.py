"""Checks of the simulator against simple theoretical cases.

Each check returns a dict with: name, expected, measured, passed.
These run inside the app ("Results and conclusion" section), from
``python scripts/generate_examples.py`` and from the pytest suite.
"""
from __future__ import annotations

import numpy as np

from .analysis import coil_face_toward_magnet, dominant_index, sweep
from .params import SimParams
from .physics import (coaxial_dflux_dX_analytic, coaxial_flux_analytic, dipole_moment_from_Bref,
                      flux_by_surface_integral, flux_per_turn, loop_field_per_amp, simulate,
                      uniform_field_ramp, MU0, trapezoid)


def _check(name, expected, measured, passed):
    return {"Check": name, "Expected": expected, "Measured": measured, "Passed": bool(passed)}


def run_all(base: SimParams | None = None) -> list[dict]:
    base = base or SimParams()
    out = []

    # 1. Loop field: off-axis elliptic formula -> on-axis limit and centre value
    a = 0.05
    Bz_c, _ = loop_field_per_amp(a, 0.0, 0.0)
    Bz_near, Br_near = loop_field_per_amp(a, 1e-6 * a, 0.03)
    Bz_axis, _ = loop_field_per_amp(a, 0.0, 0.03)
    rel = abs(Bz_near - Bz_axis) / Bz_axis
    out.append(_check("Loop field at centre = μ0/(2a) and off-axis formula → on-axis formula",
                      f"{MU0/(2*a):.6e} T/A; relative difference < 1e-6",
                      f"{float(Bz_c):.6e} T/A; difference {float(rel):.1e}",
                      abs(Bz_c - MU0 / (2 * a)) / (MU0 / (2 * a)) < 1e-12 and rel < 1e-6))

    # 2. Flux by reciprocity vs. direct surface integral of the dipole field
    m = dipole_moment_from_Bref(0.3)
    worst = 0.0
    for th in (0.0, 35.0, 60.0, 150.0):
        for X in (-0.12, -0.06, 0.08):
            f1 = float(flux_per_turn(X, m, a, np.deg2rad(th)))
            f2 = flux_by_surface_integral(X, m, a, np.deg2rad(th))
            worst = max(worst, abs(f1 - f2) / abs(f2))
    out.append(_check("Flux via reciprocity matches direct integration of B·dA over the coil",
                      "relative difference < 1e-6 (12 positions/tilts)",
                      f"largest difference {worst:.1e}", worst < 1e-6))

    # 3. Numerical dΦ/dt vs analytic chain rule (coaxial pass-through)
    res = simulate(base.with_(theta_deg=0.0, motion="pass", duration=None))
    analytic = -res.params.N * coaxial_dflux_dX_analytic(res.x, res.m, res.a) * res.vx
    moving = res.vx != 0
    inner = moving & np.roll(moving, 1) & np.roll(moving, -1)  # skip the start/stop kinks
    err = np.max(np.abs(res.emf[inner] - analytic[inner])) / np.max(np.abs(analytic))
    out.append(_check("Central-difference EMF matches analytic −N (dΦ/dx) v (θ = 0)",
                      "error < 0.1 % of peak", f"{err*100:.4f} % of peak", err < 1e-3))

    # 4. Uniform field ramp: emf = -N A cos(theta) dB/dt
    ramp = uniform_field_ramp(N=200, A=0.01, theta_deg=30.0, B0=0.1, dBdt=0.5)
    err = np.max(np.abs(ramp["emf"] - ramp["emf_theory"])) / abs(ramp["emf_theory"])
    out.append(_check("Uniform field, B = B0 + (dB/dt)t: ε = −N A cos θ dB/dt",
                      f"{ramp['emf_theory']:.6f} V (constant)",
                      f"{ramp['emf'].mean():.6f} V, max deviation {err:.1e}", err < 1e-9))

    # 5. Stationary magnet -> zero EMF
    st = simulate(base.with_(motion="stationary", duration=1.0))
    out.append(_check("Stationary magnet and coil (constant flux) give zero EMF",
                      "max |ε| = 0 V", f"max |ε| = {np.abs(st.emf).max():.2e} V; Φ = {st.flux[0]:.3e} Wb",
                      np.abs(st.emf).max() == 0.0 and abs(st.flux[0]) > 0))

    # 6-8. Proportionality: N, v, B
    for name, vals, label in (("N", [100, 200, 400], "turns N"),
                              ("v", [0.5, 1.0, 2.0], "speed v"),
                              ("B_ref", [0.1, 0.2, 0.4], "magnet strength B₀")):
        sw = sweep(base, name, vals)
        ratios = sw["peak"][1:] / sw["peak"][:-1]
        ok = np.allclose(ratios, 2.0, rtol=2e-3)
        out.append(_check(f"Doubling {label} doubles the peak EMF",
                          "ratios 2.000, 2.000",
                          ", ".join(f"{r:.4f}" for r in ratios), ok))

    # 9. Sign reversal: approach vs recede, and polarity reversal
    b0 = base.with_(theta_deg=0.0, duration=None)
    e_app = simulate(b0.with_(motion="approach", polarity="N"))
    e_rec = simulate(b0.with_(motion="recede", polarity="N"))
    e_app_s = simulate(b0.with_(motion="approach", polarity="S"))
    s1 = np.sign(e_app.emf[dominant_index(e_app.emf)])
    s2 = np.sign(e_rec.emf[dominant_index(e_rec.emf)])
    s3 = np.sign(e_app_s.emf[dominant_index(e_app_s.emf)])
    out.append(_check("EMF sign reverses when the motion reverses or the polarity reverses",
                      "approach N: −, recede N: +, approach S: +",
                      f"approach N: {'+' if s1 > 0 else '−'}, recede N: {'+' if s2 > 0 else '−'}, "
                      f"approach S: {'+' if s3 > 0 else '−'}",
                      s1 < 0 and s2 > 0 and s3 > 0))

    # 10. Lenz: approaching N pole sees an N face (repulsion)
    i = dominant_index(e_app.emf)
    face = coil_face_toward_magnet(float(e_app.current[i]), float(e_app.x[i]), 0.0)
    out.append(_check("Lenz's law: an approaching N pole induces an N face on the coil (repulsion)",
                      "coil face toward magnet = N", f"coil face toward magnet = {face}", face == "N"))

    # 11. Edge-on coil: flux stays zero, so no EMF even though the magnet moves
    eo = simulate(base.with_(theta_deg=90.0, motion="approach", duration=None))
    out.append(_check("Coil edge-on (θ = 90°): magnet moves but flux stays 0, so ε = 0",
                      "max |Φ| ≈ 0, max |ε| ≈ 0",
                      f"max |Φ| = {np.abs(eo.flux).max():.1e} Wb, max |ε| = {np.abs(eo.emf).max():.1e} V",
                      np.abs(eo.emf).max() < 1e-9 * max(1.0, np.abs(e_app.emf).max())))

    # 12. Charge: ∫I dt = −N ΔΦ / R, independent of speed
    q = []
    for v in (0.5, 2.0):
        r = simulate(b0.with_(motion="approach", v=v))
        q.append((trapezoid(r.current, r.t), -r.params.N * (r.flux[-1] - r.flux[0]) / r.params.R))
    ok = all(abs(qi - qt) / abs(qt) < 1e-3 for qi, qt in q) and abs(q[0][0] - q[1][0]) / abs(q[1][0]) < 1e-3
    out.append(_check("Total charge ∫I dt = −NΔΦ/R and does not depend on speed",
                      f"{q[0][1]:.5e} C at both speeds",
                      f"v = 0.5 m/s: {q[0][0]:.5e} C; v = 2 m/s: {q[1][0]:.5e} C", ok))

    # 13. Peak vs average EMF
    mt = e_app.metrics()
    out.append(_check("Average EMF (−NΔΦ/Δt) is smaller in magnitude than the peak EMF",
                      "|ε_avg| < |ε_peak|",
                      f"|ε_avg| = {abs(mt['average_emf_V']):.4f} V, |ε_peak| = {mt['max_abs_emf_V']:.4f} V",
                      abs(mt["average_emf_V"]) < mt["max_abs_emf_V"]))

    # 14. Coaxial closed-form flux
    Xs = np.linspace(-0.3, 0.3, 61)
    diff = np.max(np.abs(flux_per_turn(Xs, m, a, 0.0) - coaxial_flux_analytic(Xs, m, a))) / \
        np.max(np.abs(coaxial_flux_analytic(Xs, m, a)))
    out.append(_check("θ = 0 flux equals the closed form μ0 m a² / (2(a² + x²)^{3/2})",
                      "relative difference < 1e-12", f"{diff:.1e}", diff < 1e-12))
    return out
