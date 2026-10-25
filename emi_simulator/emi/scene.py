"""Geometry of the 2-D side view (all lengths in centimetres).

The page is the x-y plane; the magnet moves along the x-axis and the coil is
centred at the origin with its normal n̂ = (cos θ, sin θ) in the page.

Field lines are NOT decorative: for an ideal point dipole the field lines in
any plane containing the dipole axis satisfy r = C sin²φ exactly (φ measured
from the dipole axis). These are the field lines of the very model used to
compute the flux. The magnet's drawn size is illustrative only — the model
treats it as a point dipole at its centre.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CM = 100.0  # metres -> centimetres


@dataclass
class SceneGeometry:
    a: float            # coil radius, cm
    theta: float        # rad
    n_hat: np.ndarray
    perp: np.ndarray    # in-plane direction along the coil's face
    winding: float      # cm, axial length of the drawn winding
    mag_len: float      # cm
    mag_thick: float    # cm
    xlim: tuple
    ylim: tuple
    polarity: int       # +1: moment along +x (N pole on the right end)

    @classmethod
    def from_result(cls, res, aspect: float = 2.1) -> "SceneGeometry":
        p = res.params
        a = res.a * CM
        theta = res.theta
        n_hat = np.array([np.cos(theta), np.sin(theta)])
        perp = np.array([-np.sin(theta), np.cos(theta)])
        mag_len = 4.0
        mag_thick = float(min(2.2, 1.1 * a))
        x_lo = float(np.min(res.x)) * CM - mag_len * 0.8 - 2.0
        x_hi = float(np.max(res.x)) * CM + mag_len * 0.8 + 2.0
        x_lo = min(x_lo, -1.6 * a - 2.0)
        x_hi = max(x_hi, 1.6 * a + 4.0)
        y_half = max(1.45 * a + 1.0, (x_hi - x_lo) / (2.0 * aspect))
        return cls(a=a, theta=theta, n_hat=n_hat, perp=perp,
                   winding=float(max(0.6, 0.22 * a)), mag_len=mag_len, mag_thick=mag_thick,
                   xlim=(x_lo, x_hi), ylim=(-y_half, y_half),
                   polarity=1 if p.polarity == "N" else -1)

    # ------------------------------------------------------------------ magnet
    def magnet_halves(self, X_cm: float):
        """Closed polygons (x, y) of the N half and the S half of the bar magnet."""
        L, h = self.mag_len, self.mag_thick

        def rect(x0, x1):
            return (np.array([x0, x1, x1, x0, x0]), np.array([-h / 2, -h / 2, h / 2, h / 2, -h / 2]))

        right = rect(X_cm, X_cm + L / 2)
        left = rect(X_cm - L / 2, X_cm)
        # The moment points toward the N pole.
        if self.polarity > 0:
            return {"N": right, "S": left,
                    "N_label": (X_cm + L / 4, 0.0), "S_label": (X_cm - L / 4, 0.0)}
        return {"N": left, "S": right,
                "N_label": (X_cm - L / 4, 0.0), "S_label": (X_cm + L / 4, 0.0)}

    # ------------------------------------------------------------- field lines
    def field_lines(self, X_cm: float, n_lines: int = 7, n_pts: int = 140):
        """Exact dipole field lines r = C sin²φ around the magnet centre.

        Returns x, y (with NaN separators) in the direction of B, plus an array
        of marker sizes that is non-zero at one point per line (for arrowheads).
        """
        L, h = self.mag_len, self.mag_thick
        span = max(self.xlim[1] - self.xlim[0], 1.0)
        C_vals = np.geomspace(0.85 * L, 1.1 * span, n_lines)
        phi = np.linspace(0.03, np.pi - 0.03, n_pts)
        mid = n_pts // 2
        xs, ys, sizes = [], [], []
        for C in C_vals:
            r = C * np.sin(phi) ** 2
            for sgn in (1.0, -1.0):
                x = X_cm + r * np.cos(phi)
                y = sgn * r * np.sin(phi)
                inside = (np.abs(x - X_cm) < L / 2) & (np.abs(y) < h / 2)
                outside = (x < self.xlim[0]) | (x > self.xlim[1]) | (y < self.ylim[0]) | (y > self.ylim[1])
                x = np.where(inside | outside, np.nan, x)
                y = np.where(inside | outside, np.nan, y)
                s = np.zeros(n_pts)
                s[mid] = 9.0
                # With phi increasing, points run from the +x end to the -x end of the magnet.
                # Outside the magnet B runs from N to S, i.e. from the +x end when the
                # moment points along +x.
                if self.polarity < 0:
                    x, y, s = x[::-1], y[::-1], s[::-1]
                xs.append(np.append(x, np.nan))
                ys.append(np.append(y, np.nan))
                sizes.append(np.append(s, 0.0))
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(sizes)

    # -------------------------------------------------------------------- coil
    def coil_turns(self, N: int, max_drawn: int = 14, depth: float = 0.22):
        """Front and back halves of the drawn turns (pseudo-perspective ellipses)."""
        drawn = int(min(N, max_drawn))
        offsets = np.linspace(-self.winding / 2, self.winding / 2, drawn) if drawn > 1 else np.array([0.0])
        s = np.linspace(0, 2 * np.pi, 121)
        fx, fy, bx, by = [], [], [], []
        for u in offsets:
            pts = (u * self.n_hat[:, None] + self.a * np.cos(s)[None, :] * self.perp[:, None]
                   + depth * self.a * np.sin(s)[None, :] * self.n_hat[:, None])
            front = np.sin(s) >= 0
            for mask, X, Y in ((front, fx, fy), (~front, bx, by)):
                X.extend(np.where(mask, pts[0], np.nan).tolist() + [np.nan])
                Y.extend(np.where(mask, pts[1], np.nan).tolist() + [np.nan])
        return (np.array(fx), np.array(fy)), (np.array(bx), np.array(by)), drawn

    def flux_segment(self):
        """The coil's face seen edge-on (the area the flux passes through)."""
        e = self.a * self.perp
        return np.array([-e[0], e[0]]), np.array([-e[1], e[1]])

    def current_marker_points(self):
        """Top and bottom of the coil (where dot/cross current symbols go)."""
        top = self.a * self.perp
        return np.array([top[0], -top[0]]), np.array([top[1], -top[1]])

    def face_label_points(self, offset: float = 1.6):
        """Positions just outside the +n̂ face and the −n̂ face of the winding."""
        d = (self.winding / 2 + offset) * self.n_hat
        up = 0.55 * self.a * self.perp
        return (d + up), (-d + up)

    def normal_arrow(self, length: float | None = None):
        L = length or (self.a * 0.9 + 2.0)
        start = (self.winding / 2) * self.n_hat
        end = start + L * self.n_hat
        return np.array([start[0], end[0]]), np.array([start[1], end[1]])


def current_symbols(I: float, threshold: float):
    """Marker symbols for (top, bottom) of the coil in the side view.

    Positive current is right-handed about n̂: it comes OUT of the page at the
    top of the drawing (n̂ × perp = +z) and goes INTO the page at the bottom.
    """
    if abs(I) < threshold:
        return ["circle-open", "circle-open"], 0.0
    if I > 0:
        return ["circle-dot", "circle-x"], 1.0
    return ["circle-x", "circle-dot"], 1.0


def frame_indices(n: int, n_frames: int, must_include=()) -> np.ndarray:
    idx = np.linspace(0, n - 1, n_frames).round().astype(int)
    idx = np.unique(np.concatenate([idx, np.asarray(must_include, dtype=int)]))
    return idx
