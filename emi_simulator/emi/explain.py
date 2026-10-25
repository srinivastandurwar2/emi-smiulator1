"""Plain-language explanations that update when parameters change."""
from __future__ import annotations

import numpy as np

from .analysis import coil_face_toward_magnet, current_sense_from_left, fmt_si
from .params import SimParams


def describe_run(res) -> list[str]:
    p = res.params
    m = res.metrics()
    i = int(np.argmax(np.abs(res.emf)))
    lines = []
    if m["max_abs_emf_V"] == 0.0:
        if p.motion == "stationary":
            lines.append(f"The magnet is not moving, so the flux through each turn stays at "
                         f"{fmt_si(res.flux[0], 'Wb')}. A constant flux gives dΦ/dt = 0, so **no EMF "
                         f"is induced** — even though the coil sits in a magnetic field.")
        else:
            lines.append("The flux through the coil never changes (the coil is edge-on to the field), "
                         "so **no EMF is induced even though the magnet moves**. Induction needs a "
                         "change of flux, not just motion.")
        return lines

    lines.append(
        f"The largest EMF, **{fmt_si(res.emf[i], 'V')}**, occurs at t = {res.t[i]:.4f} s when the magnet "
        f"centre is {abs(res.x[i])*100:.1f} cm from the coil. That is where the flux "
        f"changes fastest (dΦ/dt = {fmt_si(res.dflux_dt[i], 'Wb/s')}), not where the flux is largest.")
    I = float(res.current[i])
    sense = current_sense_from_left(I, p.theta_deg)
    face = coil_face_toward_magnet(I, float(res.x[i]), p.theta_deg)
    if sense != "none" and face != "–":
        verb = "repels" if face == p.polarity else "attracts"
        lines.append(
            f"At that moment the current ({fmt_si(I, 'A')}) flows **{sense}** seen from the left. "
            f"The coil acts like a small magnet whose **{face}** face points at the magnet, so it "
            f"{verb} the magnet's {p.polarity} pole — opposing the change in flux (Lenz's law).")
    lines.append(
        f"Average EMF over the motion (−N ΔΦ/Δt) is {fmt_si(m['average_emf_V'], 'V')}, while the peak "
        f"is {fmt_si(m['max_abs_emf_V'], 'V')}: the peak depends on how quickly flux changes at one "
        f"instant, the average only on the start and end flux.")
    return lines


_LABELS = {"B_ref": "magnet strength B₀", "N": "number of turns N", "v": "speed v", "A": "coil area A",
           "theta_deg": "tilt θ", "R": "resistance R", "d_start": "starting distance",
           "d_min": "closest distance", "motion": "motion", "polarity": "polarity", "duration": "duration"}


def describe_change(prev: SimParams | None, prev_metrics: dict | None, res) -> list[str]:
    """Explain what changed since the previous run and why the result moved."""
    if prev is None or prev_metrics is None:
        return []
    p = res.params
    m = res.metrics()
    changed = [k for k in _LABELS if getattr(prev, k) != getattr(p, k)]
    if not changed:
        return []
    old, new = prev_metrics["max_abs_emf_V"], m["max_abs_emf_V"]
    ratio = (new / old) if old > 0 else float("nan")
    head = (f"You changed the **{', '.join(_LABELS[k] for k in changed)}**. Peak EMF went from "
            f"{fmt_si(old, 'V')} to {fmt_si(new, 'V')}" + (f" (×{ratio:.3f})." if np.isfinite(ratio) else "."))
    out = [head]
    if len(changed) == 1:
        k = changed[0]
        if k in ("B_ref", "N", "v"):
            x0, x1 = float(getattr(prev, k)), float(getattr(p, k))
            why = {"B_ref": "B₀ scales the magnet's dipole moment, so the field — and the flux at every "
                            "position — scales by the same factor. The shape of the field is unchanged.",
                   "N": "every turn links the same flux, so the flux linkage NΦ scales with N. "
                        "(In a real coil extra turns also add resistance; here R is held fixed.)",
                   "v": "the magnet follows the same path, so the same flux change happens in less "
                        "(or more) time: dΦ/dt = (dΦ/dx)·v."}[k]
            out.append(f"The parameter changed by ×{x1/x0:.3f} and the peak EMF by ×{ratio:.3f}: "
                       f"directly proportional, because {why}")
        elif k == "R":
            out.append("The EMF does not depend on R; only the current I = ε/R changes.")
        elif k == "polarity":
            out.append("Reversing the magnet reverses the flux, so the EMF and current reverse sign "
                       "while their sizes stay the same.")
        elif k == "theta_deg":
            out.append("Only the field component along the coil's normal contributes to the flux. "
                       "Tilting the coil toward 90° reduces that component (roughly as cos θ near the axis).")
        elif k == "A":
            out.append("A bigger coil encloses more field lines, but because the magnet's field weakens "
                       "away from its axis the EMF does not grow exactly in proportion to A.")
        elif k == "d_min":
            out.append("The dipole field falls off roughly as 1/r³, so letting the magnet come closer "
                       "makes the flux change much faster near the end of the motion.")
        elif k == "motion":
            out.append("The motion determines when the flux rises or falls; the EMF sign follows "
                       "the sign of the flux change.")
    return out
