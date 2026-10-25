"""Physics engine for the electromagnetic-induction simulator.

Model summary
-------------
* The bar magnet is modelled as an ideal **point magnetic dipole** of moment
  ``m`` (A*m^2) located at the magnet's centre and pointing along the lab
  x-axis. The magnet always moves along the x-axis.
* The coil is a **thin flat coil** of ``N`` circular turns of radius
  ``a = sqrt(A/pi)``, centred at the origin. Its normal (area vector direction)
  makes an angle ``theta`` with the x-axis (the magnet's axis).
* Every turn links the same flux ``Phi`` (thin-coil assumption), so the flux
  linkage is ``N * Phi``.
* The flux of a dipole through a circular loop is computed **exactly** with the
  reciprocity theorem: ``Phi = m_vec . B_loop(r_magnet) / I`` where
  ``B_loop`` is the field the same loop would create at the magnet's position
  if it carried current ``I``. ``B_loop`` is evaluated with the standard
  closed-form expressions containing complete elliptic integrals, so the flux
  is correct for any position and tilt (not only on the axis).
* EMF: ``emf(t) = -N dPhi/dt`` using second-order central finite differences
  (``numpy.gradient``).
* Current: ``I(t) = emf(t) / R`` (purely resistive circuit, coil inductance and
  the force on the magnet are neglected).

Sign convention
---------------
Positive flux is flux along the coil's unit normal ``n_hat``. Positive EMF and
positive current circulate around ``n_hat`` according to the right-hand rule
(curl the fingers of your right hand in the direction of the current; your
thumb points along ``n_hat``). With this convention ``emf = -N dPhi/dt``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ellipe, ellipk

from .params import R_REF, SimParams, validate

MU0 = 4e-7 * np.pi  # T*m/A (vacuum permeability)

# numpy >= 2.0 renamed trapz -> trapezoid; support both.
trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# ---------------------------------------------------------------------------
# Magnet description
# ---------------------------------------------------------------------------
def dipole_moment_from_Bref(B_ref: float, r_ref: float = R_REF) -> float:
    """Dipole moment (A*m^2) whose on-axis field at distance r_ref equals B_ref.

    On the axis of a dipole, B = mu0 * m / (2 pi r^3).
    """
    return 2.0 * np.pi * r_ref**3 * B_ref / MU0


def dipole_axial_field(m: float, r) -> np.ndarray:
    """Magnitude of a dipole's field on its own axis at distance r (m)."""
    r = np.asarray(r, dtype=float)
    return MU0 * m / (2.0 * np.pi * r**3)


def dipole_field(m_vec, r_vec) -> np.ndarray:
    """Field of a point dipole m_vec at displacement(s) r_vec (shape (..., 3))."""
    r_vec = np.asarray(r_vec, dtype=float)
    m_vec = np.asarray(m_vec, dtype=float)
    r = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r_hat = r_vec / r
    m_dot_r = np.sum(m_vec * r_hat, axis=-1, keepdims=True)
    return MU0 / (4.0 * np.pi) * (3.0 * m_dot_r * r_hat - m_vec) / r**3


# ---------------------------------------------------------------------------
# Field of a circular current loop (per ampere), exact closed form
# ---------------------------------------------------------------------------
def loop_field_per_amp(a: float, rho, z):
    """Axial and radial field (T per A) of a circular loop of radius ``a``.

    The loop lies in the plane z = 0 and is centred on the z-axis. ``rho`` is
    the distance from the axis and ``z`` the axial coordinate of the field
    point. Uses the complete elliptic integrals K(m), E(m) with parameter
    m = k^2 = 4 a rho / ((a + rho)^2 + z^2).
    """
    rho, z = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(z, dtype=float))
    rho = np.abs(rho)
    Bz = np.empty_like(rho)
    Br = np.zeros_like(rho)

    on_axis = rho < 1e-9 * a
    Bz[on_axis] = MU0 * a**2 / (2.0 * (a**2 + z[on_axis] ** 2) ** 1.5)

    off = ~on_axis
    if np.any(off):
        r = rho[off]
        zz = z[off]
        Q = (a + r) ** 2 + zz**2
        m = np.minimum(4.0 * a * r / Q, 1.0 - 1e-15)
        K = ellipk(m)
        E = ellipe(m)
        # tiny regulariser: the field is genuinely singular only on the wire
        D = (a - r) ** 2 + zz**2 + (1e-9 * a) ** 2
        pref = MU0 / (2.0 * np.pi * np.sqrt(Q))
        Bz[off] = pref * (K + (a * a - r * r - zz * zz) / D * E)
        Br[off] = pref * (zz / r) * (-K + (a * a + r * r + zz * zz) / D * E)
    return Bz, Br


# ---------------------------------------------------------------------------
# Flux through the tilted coil
# ---------------------------------------------------------------------------
def flux_per_turn(X, m_signed: float, a: float, theta_rad: float) -> np.ndarray:
    """Magnetic flux (Wb) through ONE turn for a dipole at (X, 0, 0).

    ``m_signed`` is the dipole moment along +x (negative if the moment points
    along -x). The coil is centred at the origin; its normal is
    n_hat = (cos theta, sin theta, 0).

    Reciprocity: Phi = m_vec . B_loop(r_mag) / I.
    In the coil's own frame the magnet sits at axial coordinate z' = X cos(theta)
    and radial distance rho' = |X sin(theta)|.
    """
    X = np.asarray(X, dtype=float)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    # Snap round-off so that 90 deg / 180 deg are exact (cos(pi/2) is 6e-17 in floating point)
    c = 0.0 if abs(c) < 1e-12 else c
    s = 0.0 if abs(s) < 1e-12 else s
    z = X * c
    rho = np.abs(X * s)
    Bz, Br = loop_field_per_amp(a, rho, z)
    # x-component of the radial unit vector rho_hat = sign(X) * |sin theta|
    return m_signed * (Bz * c + Br * np.sign(X) * abs(s))


def flux_by_surface_integral(X: float, m_signed: float, a: float, theta_rad: float,
                             n_r: int = 200, n_phi: int = 240) -> float:
    """Independent check: integrate B_dipole . n over the coil's disc numerically.

    Only valid when the dipole is not on the disc itself. Used by the
    validation suite to confirm the reciprocity/elliptic-integral result.
    """
    n_hat = np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0])
    u = np.array([-np.sin(theta_rad), np.cos(theta_rad), 0.0])  # in-plane, in page
    w = np.array([0.0, 0.0, 1.0])                               # in-plane, out of page
    # Gauss-Legendre in r, uniform (spectrally accurate) in phi
    xg, wg = np.polynomial.legendre.leggauss(n_r)
    r = 0.5 * a * (xg + 1.0)
    wr = 0.5 * a * wg
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    wphi = 2.0 * np.pi / n_phi
    R, P = np.meshgrid(r, phi, indexing="ij")
    pts = (R[..., None] * (np.cos(P)[..., None] * u + np.sin(P)[..., None] * w))
    r_vec = pts - np.array([X, 0.0, 0.0])
    B = dipole_field(np.array([m_signed, 0.0, 0.0]), r_vec)
    Bn = B @ n_hat
    return float(np.sum(Bn * R * wr[:, None]) * wphi)


def coaxial_flux_analytic(X, m_signed: float, a: float) -> np.ndarray:
    """Closed form for theta = 0: Phi = mu0 m a^2 / (2 (a^2 + X^2)^(3/2))."""
    X = np.asarray(X, dtype=float)
    return MU0 * m_signed * a**2 / (2.0 * (a**2 + X**2) ** 1.5)


def coaxial_dflux_dX_analytic(X, m_signed: float, a: float) -> np.ndarray:
    """d(Phi)/dX for the coaxial case (used to validate the numerical derivative)."""
    X = np.asarray(X, dtype=float)
    return -1.5 * MU0 * m_signed * a**2 * X / (a**2 + X**2) ** 2.5


# ---------------------------------------------------------------------------
# Trajectory of the magnet
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    t: np.ndarray            # s
    x: np.ndarray            # m, magnet-centre position (coil at x = 0)
    vx: np.ndarray           # m/s, analytic velocity
    t_start: float           # s, motion begins
    t_move: float            # s, duration of the motion
    duration: float          # s, total simulated time
    truncated: bool          # True if the duration cut the motion short


def build_trajectory(p: SimParams) -> Trajectory:
    """Piecewise constant-speed motion along the x-axis.

    The magnet always starts on the left of the coil (x < 0). It rests for a
    short lead-in (10 % of the motion time), moves at constant speed ``v``,
    then rests again. The rest periods show that a stationary magnet produces
    no EMF.
    """
    v = p.v
    if p.motion == "pass":
        x0, legs = -p.d_start, [(+1, 2.0 * p.d_start)]
    elif p.motion == "approach":
        x0, legs = -p.d_start, [(+1, p.d_start - p.d_min)]
    elif p.motion == "recede":
        x0, legs = -p.d_min, [(-1, p.d_start - p.d_min)]
    elif p.motion == "in_out":
        L = p.d_start - p.d_min
        x0, legs = -p.d_start, [(+1, L), (-1, L)]
    else:  # stationary: magnet held at the closest distance
        x0, legs = -p.d_min, []

    t_move = sum(length for _, length in legs) / v
    if t_move > 0:
        t_start = 0.10 * t_move
        auto_duration = t_start + t_move + 0.15 * t_move
    else:
        t_start = 0.0
        auto_duration = 1.0
    duration = auto_duration if p.duration is None else float(p.duration)

    t = np.linspace(0.0, duration, p.n_samples)
    x = np.full_like(t, x0)
    vx = np.zeros_like(t)
    t_leg0 = t_start
    pos = x0
    for direction, length in legs:
        t_leg = length / v
        tau = np.clip(t - t_leg0, 0.0, t_leg)
        moving = (t >= t_leg0) & (t < t_leg0 + t_leg)
        x = np.where(t >= t_leg0, pos + direction * v * tau, x)
        vx = np.where(moving, direction * v, vx)
        pos = pos + direction * length
        t_leg0 += t_leg

    truncated = duration < t_start + t_move - 1e-12
    return Trajectory(t=t, x=x, vx=vx, t_start=t_start, t_move=t_move,
                      duration=duration, truncated=truncated)


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    params: SimParams
    t: np.ndarray
    x: np.ndarray
    vx: np.ndarray
    flux: np.ndarray          # Wb, per turn
    dflux_dt: np.ndarray      # Wb/s, per turn
    emf: np.ndarray           # V
    current: np.ndarray       # A
    m: float                  # A m^2 (signed, along +x)
    a: float                  # m, coil radius
    theta: float              # rad
    t_start: float
    t_move: float
    duration: float
    messages: list = field(default_factory=list)

    # ---- summary quantities -------------------------------------------------
    def metrics(self) -> dict:
        p = self.params
        emf, I = self.emf, self.current
        peak_pos = float(max(emf.max(), 0.0))
        peak_neg = float(min(emf.min(), 0.0))
        max_abs = float(np.abs(emf).max())
        # Flux change during the motion (initial -> final position)
        dphi = float(self.flux[-1] - self.flux[0])
        # Net change of flux linkage divided by motion time (Faraday "average EMF")
        t_motion = min(self.t_move, max(self.duration - self.t_start, 0.0))
        avg_emf = float(-p.N * dphi / t_motion) + 0.0 if t_motion > 0 else 0.0  # +0.0 avoids "-0"
        # Time-average of |emf| while the magnet moves (rectified average)
        moving = (self.t >= self.t_start) & (self.t <= self.t_start + t_motion)
        mean_abs = float(trapezoid(np.abs(emf[moving]), self.t[moving]) / t_motion) if t_motion > 0 and moving.sum() > 2 else 0.0
        charge = float(trapezoid(I, self.t))
        return {
            "peak_pos_emf_V": peak_pos,
            "peak_neg_emf_V": peak_neg,
            "max_abs_emf_V": max_abs,
            "max_abs_current_A": float(np.abs(I).max()),
            "flux_initial_Wb": float(self.flux[0]),
            "flux_final_Wb": float(self.flux[-1]),
            "max_abs_flux_Wb": float(np.abs(self.flux).max()),
            "delta_flux_Wb": dphi,
            "delta_linkage_Wb_turns": p.N * dphi,
            "average_emf_V": avg_emf,
            "mean_abs_emf_V": mean_abs,
            "net_charge_C": charge,
            "motion_time_s": self.t_move,
            "dipole_moment_Am2": abs(self.m),
            "coil_radius_m": self.a,
            "mean_normal_field_at_max_flux_T": float(np.abs(self.flux).max() / p.A),
            "time_of_max_abs_emf_s": float(self.t[int(np.argmax(np.abs(emf)))]),
        }

    def sample_at(self, t_query: float) -> dict:
        """Instantaneous values at the time nearest to t_query."""
        i = int(np.clip(np.searchsorted(self.t, t_query), 0, len(self.t) - 1))
        return {"t": float(self.t[i]), "x": float(self.x[i]), "flux": float(self.flux[i]),
                "emf": float(self.emf[i]), "current": float(self.current[i]), "index": i}


def simulate(p: SimParams) -> SimResult:
    """Run one complete simulation. Input is validated (clamped) first."""
    p, msgs = validate(p)
    traj = build_trajectory(p)
    m = dipole_moment_from_Bref(p.B_ref) * (1.0 if p.polarity == "N" else -1.0)
    a = float(np.sqrt(p.A / np.pi))
    theta = np.deg2rad(p.theta_deg)

    flux = flux_per_turn(traj.x, m, a, theta)
    # Uniform time grid -> textbook central difference (Phi[i+1] - Phi[i-1]) / (2 dt) in the
    # interior and second-order one-sided differences at the two end points.
    dt = traj.t[1] - traj.t[0]
    dflux_dt = np.gradient(flux, dt, edge_order=2)
    # Exactly constant flux -> exactly zero derivative (removes round-off noise)
    if np.ptp(flux) == 0.0:
        dflux_dt = np.zeros_like(flux)
    emf = -p.N * dflux_dt
    current = emf / p.R

    if traj.truncated:
        msgs.append("The chosen duration ends before the motion is complete, so only part "
                    "of the trajectory is simulated.")
    if p.motion == "pass":
        clearance = a * abs(np.cos(theta))
        if clearance < 0.01:
            msgs.append(
                f"At θ = {p.theta_deg:.0f}° the magnet's path passes within {clearance*100:.1f} cm "
                "of the coil wire. A real magnet would hit the windings, and the point-dipole "
                "model becomes unrealistic near the wire. Use a smaller tilt for pass-through.")
    if 89.5 <= p.theta_deg <= 90.5:
        msgs.append("At θ ≈ 90° the coil plane contains the magnet's axis, so the flux through "
                    "the coil stays (almost) zero and almost no EMF is induced.")

    return SimResult(params=p, t=traj.t, x=traj.x, vx=traj.vx, flux=flux, dflux_dt=dflux_dt,
                     emf=emf, current=current, m=m, a=a, theta=theta, t_start=traj.t_start,
                     t_move=traj.t_move, duration=traj.duration, messages=msgs)


# ---------------------------------------------------------------------------
# Introductory model: uniform field changing with time (no motion)
# ---------------------------------------------------------------------------
def uniform_field_ramp(N: int, A: float, theta_deg: float, B0: float, dBdt: float,
                       duration: float = 1.0, n: int = 1001, R: float = 10.0) -> dict:
    """Phi = B(t) A cos(theta) with B(t) = B0 + (dB/dt) t.

    Faraday's law predicts emf = -N A cos(theta) dB/dt (a constant).
    """
    t = np.linspace(0.0, duration, n)
    B = B0 + dBdt * t
    flux = B * A * np.cos(np.deg2rad(theta_deg))
    emf = -N * np.gradient(flux, t[1] - t[0], edge_order=2)
    return {"t": t, "B": B, "flux": flux, "emf": emf, "current": emf / R,
            "emf_theory": -N * A * np.cos(np.deg2rad(theta_deg)) * dBdt}
