"""Run with:  python -m pytest -q   (from the project folder)."""
import numpy as np
import pytest

from emi.analysis import fmt_si, lenz_experiment, proportional_fit, sweep
from emi.params import RANGES, SimParams, validate
from emi.physics import (coaxial_flux_analytic, dipole_moment_from_Bref, flux_by_surface_integral,
                         flux_per_turn, simulate, uniform_field_ramp)
from emi.validation import run_all


@pytest.mark.parametrize("check", run_all(), ids=lambda c: c["Check"][:50])
def test_validation_suite(check):
    assert check["Passed"], f"{check['Check']}: expected {check['Expected']}, got {check['Measured']}"


@pytest.mark.parametrize("theta", [0.0, 20.0, 45.0, 75.0, 110.0, 180.0])
@pytest.mark.parametrize("X", [-0.15, -0.07, 0.09])
def test_flux_reciprocity_vs_surface_integral(theta, X):
    m = dipole_moment_from_Bref(0.5)
    a = 0.04
    f1 = float(flux_per_turn(X, m, a, np.deg2rad(theta)))
    f2 = flux_by_surface_integral(X, m, a, np.deg2rad(theta))
    assert f1 == pytest.approx(f2, rel=1e-8, abs=1e-15)


def test_theta_180_reverses_flux():
    m, a = dipole_moment_from_Bref(0.3), 0.04
    X = np.linspace(-0.2, 0.2, 41)
    assert np.allclose(flux_per_turn(X, m, a, np.pi), -coaxial_flux_analytic(X, m, a), rtol=1e-12)


@pytest.mark.parametrize("motion", ["pass", "in_out", "approach", "recede", "stationary"])
@pytest.mark.parametrize("polarity", ["N", "S"])
def test_all_modes_run_and_are_finite(motion, polarity):
    r = simulate(SimParams(motion=motion, polarity=polarity, theta_deg=30))
    for arr in (r.flux, r.emf, r.current):
        assert np.all(np.isfinite(arr))
    m = r.metrics()
    assert m["peak_pos_emf_V"] >= 0 >= m["peak_neg_emf_V"]
    assert m["max_abs_current_A"] == pytest.approx(m["max_abs_emf_V"] / r.params.R)


def test_rest_periods_have_zero_emf():
    r = simulate(SimParams(motion="approach"))
    before = r.t < r.t_start - 2 * (r.t[1] - r.t[0])
    after = r.t > r.t_start + r.t_move + 2 * (r.t[1] - r.t[0])
    tol = 1e-12 * np.abs(r.emf).max()  # round-off level only
    assert np.abs(r.emf[before]).max() <= tol and np.abs(r.emf[after]).max() <= tol


def test_pass_through_is_antisymmetric():
    r = simulate(SimParams(motion="pass"))
    m = r.metrics()
    assert m["peak_pos_emf_V"] == pytest.approx(-m["peak_neg_emf_V"], rel=1e-3)
    assert abs(m["delta_flux_Wb"]) < 1e-15


def test_resistance_changes_current_not_emf():
    r1 = simulate(SimParams(R=5))
    r2 = simulate(SimParams(R=50))
    assert np.allclose(r1.emf, r2.emf)
    assert np.allclose(r1.current, 10 * r2.current)


def test_uniform_ramp_matches_theory():
    out = uniform_field_ramp(N=500, A=0.02, theta_deg=60, B0=0.2, dBdt=-1.5)
    assert np.allclose(out["emf"], out["emf_theory"], rtol=1e-10)
    assert out["emf_theory"] == pytest.approx(-500 * 0.02 * 0.5 * -1.5)


@pytest.mark.parametrize("name,vals", [("N", [10, 250, 1000]), ("v", [0.01, 0.7, 5.0]), ("B_ref", [0.01, 0.37, 1.0])])
def test_proportionality_over_full_range(name, vals):
    sw = sweep(SimParams(motion="in_out"), name, vals)
    ratios = sw["peak"] / sw["x"]
    assert np.ptp(ratios) / ratios.mean() < 2e-3
    assert sw["fit"]["r2"] > 0.99999


def test_lenz_cases():
    lz = lenz_experiment(SimParams())
    faces = {c["Case"]: c["Coil face toward magnet"] for c in lz["cases"]}
    assert faces == {"N pole approaches": "N", "N pole moves away": "S",
                     "S pole approaches": "S", "S pole moves away": "N"}
    assert np.abs(lz["edge_on"].emf).max() == 0.0


def test_validate_clamps_bad_input():
    p, msgs = validate(SimParams(B_ref=5, N=-3, v=float("nan"), A=0, theta_deg=400, d_start=0.1,
                                 d_min=0.2, motion="approach", polarity="X", R=0))
    assert RANGES["B_ref"][0] <= p.B_ref <= RANGES["B_ref"][1]
    assert p.N == 10 and p.v == SimParams().v and p.polarity == "N"
    assert p.d_min < p.d_start and p.R > 0
    assert len(msgs) >= 7


def test_manual_duration_longer_and_truncated():
    long = simulate(SimParams(motion="approach", duration=5.0))
    assert long.duration == 5.0 and long.emf[-1] == pytest.approx(0, abs=1e-12)
    short = simulate(SimParams(motion="approach", duration=0.05))
    assert any("ends before the motion" in msg for msg in short.messages)


def test_proportional_fit_and_formatting():
    fit = proportional_fit([1, 2, 3], [2, 4, 6])
    assert fit["k"] == pytest.approx(2) and fit["r2"] == pytest.approx(1)
    assert fmt_si(0.00123, "V") == "1.23 mV" and fmt_si(0, "A") == "0 A"
