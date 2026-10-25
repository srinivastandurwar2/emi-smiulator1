# Theory, model and numerical method

## 1. Faraday's law and Lenz's law

For a coil of N turns, each linking magnetic flux Φ_B, the induced EMF is

    ε = −N dΦ_B/dt

The flux through one turn is Φ_B = ∫ **B**·d**A**. For a uniform field at angle θ to the coil's
area vector this becomes Φ_B = BA cos θ, and if only B changes with time

    ε = −N A cos θ dB/dt.

**Lenz's law** is the minus sign: the induced current flows so that its own magnetic field opposes
the *change* of flux. An increasing flux is opposed by an induced field pointing the other way; a
decreasing flux is "supported" by an induced field pointing the same way.

With a resistive circuit of total resistance R, the induced current is I(t) = ε(t)/R.

## 2. Why the simple formula Φ = BA cos θ is not enough here

A bar magnet's field is strongly non-uniform: it falls off roughly as 1/r³ and its direction
changes from point to point. As the magnet moves, the field at the coil changes in both size and
shape. The simulator therefore calculates the flux from a position-dependent model instead of
assuming a constant B.

## 3. The magnet: a point magnetic dipole

The magnet is represented by a point dipole of moment **m** at its centre, pointing along the
x-axis (towards its N pole). Its field is

    B(r) = (μ₀/4π) [3(m·r̂) r̂ − m] / r³,

and on its own axis B = μ₀ m / (2π r³).

**The control B₀** is defined as the field on the magnet's axis at r_ref = 2 cm from its centre:

    m = 2π r_ref³ B₀ / μ₀.

So B₀ = 0.3 T means m = 12 A·m² (a large neodymium magnet). This keeps three things distinct:

| Quantity | Meaning | Changes when… |
|---|---|---|
| Magnet strength B₀ (control) | property of the magnet | you move the B₀ slider |
| Field at the coil | depends on B₀ and on where the magnet is | the magnet moves, or B₀ changes |
| Field distribution | the shape of the field-line pattern | never (fixed by the dipole model) |

When B₀ is varied, the coil (N, A, θ), the speed, the path and R are held constant, and the field
everywhere scales by the same factor.

## 4. The coil and the flux

The coil is N thin circular turns of radius a = √(A/π) centred at the origin, with unit normal
n̂ = (cos θ, sin θ, 0). All turns are assumed to link the same flux (thin-coil assumption).

### Exact flux by reciprocity
The mutual inductance between a dipole and a loop is symmetric, which gives

    Φ = m · B_loop(r_magnet) / I,

where B_loop is the field the loop would produce at the magnet's position if it carried current I.
In the coil's own frame the magnet sits at axial coordinate z = X cos θ and radial distance
ρ = |X sin θ|. The loop field has the closed form (Jackson §5.5; Simpson et al., NASA TM)

    k² = 4aρ / [(a+ρ)² + z²]
    B_z = μ₀I / (2π√((a+ρ)²+z²)) · [ K(k²) + (a² − ρ² − z²)/((a−ρ)² + z²) · E(k²) ]
    B_ρ = μ₀I z / (2πρ√((a+ρ)²+z²)) · [ −K(k²) + (a² + ρ² + z²)/((a−ρ)² + z²) · E(k²) ]

with K and E the complete elliptic integrals of the first and second kind. Then

    Φ = m [ B_z cos θ + B_ρ sign(X) |sin θ| ] / I.

For θ = 0 this reduces to the familiar result

    Φ(x) = μ₀ m a² / [2 (a² + x²)^{3/2}].

**Validation:** the reciprocity result was compared with a direct numerical surface integral of
the dipole field over the coil's disc (Gauss–Legendre quadrature) at many positions and tilts;
the two agree to about 10⁻¹⁵ relative error.

### Field lines in the animation
For an ideal dipole the field lines in a plane containing its axis are exactly r = C sin²φ
(φ measured from the axis). These are the lines drawn, so they belong to the same model used for
the calculation. The coil's own (tiny) induced field is not drawn. The drawn magnet is 4 cm long
for visibility; in the calculation it is a point.

## 5. Motion of the magnet

The magnet starts on the left of the coil (x < 0), rests for 10 % of the motion time, moves at
constant speed v, then rests again (the rests show that no motion → no EMF). Modes:

| Mode | Path |
|---|---|
| Pass through | −d₀ → +d₀ |
| Push in, pull out | −d₀ → −d_min → −d₀ |
| Approach | −d₀ → −d_min |
| Move away | −d_min → −d₀ |
| Stationary | held at −d_min |

Since the path is fixed, Φ depends only on position, and by the chain rule

    dΦ/dt = (dΦ/dx) · v,

which is why the EMF at each point of the path is proportional to v.

## 6. Numerical method

Φ is evaluated at n = 3001 equally spaced times. The rate of change is obtained with the
second-order central difference

    (dΦ/dt)_i ≈ (Φ_{i+1} − Φ_{i−1}) / (2Δt),

and second-order one-sided differences at the two end points (`numpy.gradient`, `edge_order=2`).
Then ε_i = −N (dΦ/dt)_i and I_i = ε_i / R. The error compared with the analytic derivative is
below 0.01 % of the peak for the default settings.

## 7. Peak and average EMF

* **Peak EMF**: the largest instantaneous |ε| — set by how fast flux changes at one moment.
* **Average EMF**: −N ΔΦ / Δt over the whole motion — depends only on the start and end flux. For
  pass-through or push-in/pull-out motion ΔΦ = 0, so the average is zero although the peak is not.
* **Net charge**: ∫ I dt = −N ΔΦ / R — independent of speed.

## 8. Sign convention

Positive flux is along n̂. Positive EMF and current circulate around n̂ by the right-hand rule
(thumb along n̂, fingers show the current). At θ = 0, n̂ points away from the magnet's starting
side, so a positive current looks **clockwise** when viewed from the magnet's starting side. In the
side view, positive current comes out of the page (⊙) at the top of the coil and goes into the page
(⊗) at the bottom.

## 9. Assumptions and limitations

1. Point dipole: accurate when the magnet is small compared with its distance from the coil;
   approximate when the magnet is inside or very close to the coil. A large tilt in pass-through
   mode makes the path pass close to the wire, where a point dipole is unrealistic (the app warns).
2. Thin coil: all turns link the same flux. A long solenoid would need the flux summed turn by turn.
3. Resistive circuit: coil self-inductance L is neglected (valid when the L/R time is much shorter
   than the pulse duration). The magnetic force on the magnet is neglected, so the speed stays
   constant.
4. Instantaneous start, stop and turnaround of the motion; real hands accelerate smoothly.
5. Vacuum/air around the coil (no iron core).

## 10. Validation checks (run automatically in the app)

1. Loop field at the centre equals μ₀/(2a); the off-axis formula tends to the on-axis formula.
2. Reciprocity flux equals direct integration of **B**·d**A**.
3. Numerical ε equals the analytic −N(dΦ/dx)v.
4. Uniform ramp: ε = −NA cos θ dB/dt.
5. Stationary magnet: ε = 0 exactly.
6–8. Doubling N, v or B₀ doubles the peak EMF.
9. EMF reverses with the direction of motion and with the polarity.
10. Approaching N pole induces an N face (repulsion).
11. Edge-on coil: magnet moves, flux and EMF stay zero.
12. ∫I dt = −NΔΦ/R at different speeds.
13. |average EMF| < |peak EMF|.
14. θ = 0 flux equals the closed-form expression.
