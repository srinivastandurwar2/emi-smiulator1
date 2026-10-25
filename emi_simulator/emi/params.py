"""Simulation parameters, allowed ranges, defaults and input validation.

All quantities are stored in SI units (tesla, metres, square metres, seconds,
ohms). The user interface may display centimetres or degrees, but converts
before building a :class:`SimParams` object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional

# Human-readable names for the motion modes (keys are used internally).
MOTION_MODES = {
    "pass": "Pass through the coil (approach, then move away)",
    "in_out": "Push in, then pull back out",
    "approach": "Approach the coil (then stop)",
    "recede": "Move away from the coil",
    "stationary": "Stationary magnet (no motion)",
}

POLARITIES = {
    "N": "North pole facing the coil",
    "S": "South pole facing the coil",
}

# (min, max) for every numeric parameter, in SI units.
RANGES = {
    "B_ref": (0.01, 1.0),        # T
    "N": (10, 1000),             # turns
    "v": (0.01, 5.0),            # m/s
    "A": (0.001, 0.1),           # m^2
    "theta_deg": (0.0, 180.0),   # degrees
    "d_start": (0.05, 0.50),     # m
    "d_min": (0.01, 0.20),       # m
    "R": (0.1, 10000.0),         # ohm
    "duration": (0.01, 500.0),   # s
}

# Distance (from the magnet's centre, along its axis) at which B_ref is defined.
R_REF = 0.02  # m


@dataclass(frozen=True)
class SimParams:
    """All inputs of one simulation run (SI units)."""

    B_ref: float = 0.30          # T, magnet's on-axis field 2 cm from its centre
    N: int = 200                 # number of turns
    v: float = 1.0               # m/s, magnet speed
    A: float = 0.005             # m^2, coil area
    theta_deg: float = 0.0       # deg, angle between magnet axis and coil normal
    d_start: float = 0.30        # m, starting distance magnet centre -> coil centre
    d_min: float = 0.03          # m, closest distance (not used for "pass")
    motion: str = "pass"
    polarity: str = "N"
    R: float = 10.0              # ohm, total circuit resistance
    duration: Optional[float] = None   # s; None means "automatic"
    n_samples: int = 3001        # number of time samples

    def as_dict(self) -> dict:
        return asdict(self)

    def with_(self, **changes) -> "SimParams":
        return replace(self, **changes)


DEFAULTS = SimParams()


def validate(p: SimParams) -> tuple[SimParams, list[str]]:
    """Clamp every parameter into its allowed range.

    Returns the corrected parameters and a list of human-readable messages
    describing anything that had to be changed. Invalid types or NaN values are
    replaced by defaults, so the physics code never receives impossible input.
    """
    msgs: list[str] = []
    fixed = {}

    def _num(name, value, default):
        try:
            x = float(value)
        except (TypeError, ValueError):
            msgs.append(f"{name}: '{value}' is not a number; using default {default}.")
            return float(default)
        if x != x or x in (float("inf"), float("-inf")):
            msgs.append(f"{name}: value is not a finite number; using default {default}.")
            return float(default)
        lo, hi = RANGES[name]
        if x < lo or x > hi:
            msgs.append(f"{name}: {x:g} is outside {lo:g} to {hi:g}; clamped.")
            x = min(max(x, lo), hi)
        return x

    for name in ("B_ref", "v", "A", "theta_deg", "d_start", "d_min", "R"):
        fixed[name] = _num(name, getattr(p, name), getattr(DEFAULTS, name))
    fixed["N"] = int(round(_num("N", p.N, DEFAULTS.N)))

    if p.duration is None:
        fixed["duration"] = None
    else:
        fixed["duration"] = _num("duration", p.duration, 1.0)

    motion = p.motion if p.motion in MOTION_MODES else DEFAULTS.motion
    if motion != p.motion:
        msgs.append(f"Unknown motion mode '{p.motion}'; using '{motion}'.")
    polarity = p.polarity if p.polarity in POLARITIES else DEFAULTS.polarity
    if polarity != p.polarity:
        msgs.append(f"Unknown polarity '{p.polarity}'; using '{polarity}'.")

    # The closest distance must be smaller than the starting distance.
    if motion in ("approach", "recede", "in_out") and fixed["d_min"] >= fixed["d_start"] - 0.005:
        new_min = max(RANGES["d_min"][0], fixed["d_start"] - 0.02)
        msgs.append(
            f"Closest distance ({fixed['d_min']*100:.1f} cm) must be less than the starting "
            f"distance ({fixed['d_start']*100:.1f} cm); using {new_min*100:.1f} cm."
        )
        fixed["d_min"] = new_min

    try:
        n_samples = int(p.n_samples)
    except (TypeError, ValueError):
        n_samples = DEFAULTS.n_samples
    n_samples = min(max(n_samples, 201), 20001)
    if n_samples % 2 == 0:
        n_samples += 1

    return SimParams(motion=motion, polarity=polarity, n_samples=n_samples, **fixed), msgs
