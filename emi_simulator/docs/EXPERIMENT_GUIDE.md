# Experiment guide

Start the app (`streamlit run app.py`), press **Reset to defaults** in the sidebar, and keep the
default settings unless a step says otherwise. Example results below were produced by
`scripts/generate_examples.py` with the default settings (B₀ = 0.3 T, N = 200, v = 1 m/s,
A = 0.005 m², θ = 0°, pass-through from 30 cm, N pole facing, R = 10 Ω).

---

## Warm-up: watch one run

1. Open **Interactive simulation** and let the animation play.
2. Note that the voltmeter reads zero while the magnet is far away or at rest.
3. As the magnet enters the coil the EMF goes negative; as it leaves, positive. Pause near each
   peak and note the current symbols (⊙ ⊗), the coil's induced poles and the galvanometer.
4. Compare the **Flux vs time** and **EMF vs time** graphs: the EMF is largest where the flux curve
   is steepest, and zero where the flux is at its maximum (magnet at the centre of the coil).

Expected: peak EMF ≈ ±0.81 V, peak current ≈ ±81 mA, maximum flux per turn ≈ 189 µWb.

---

## Experiment A — effect of magnet strength

**Aim.** Study how the peak EMF depends on the magnet strength B₀.
**Held constant.** N, A, θ, v, path, R.

1. Open **Virtual experiments → A · Magnet strength**.
2. Keep the values `0.1, 0.2, 0.4, 0.6, 0.8, 1.0` (or enter your own) and press **Run experiment A**.
3. Copy the table into your record and look at the ratio |EMF|/B₀.

| S.No. | B₀ (T) | Peak \|ε\| (V) | \|ε\|/B₀ (V/T) |
|---|---|---|---|
| 1 | 0.1 | 0.2712 | 2.712 |
| 2 | 0.2 | 0.5423 | 2.712 |
| 3 | 0.4 | 1.0847 | 2.712 |
| 4 | 0.6 | 1.6270 | 2.712 |
| 5 | 0.8 | 2.1694 | 2.712 |
| 6 | 1.0 | 2.7117 | 2.712 |

**Result.** Peak EMF ∝ B₀ (straight line through the origin).
**Reason.** B₀ scales the field everywhere, so Φ(x) and dΦ/dt scale by the same factor.

---

## Experiment B — effect of the number of turns

**Aim.** Show that the EMF is proportional to N. **Held constant.** B₀ (so the flux per turn), A,
θ, v, path, R.

1. **Virtual experiments → B · Number of turns**; values `50, 100, 200, 400, 600, 800, 1000`.
2. Press **Run experiment B**.

| S.No. | N | Peak \|ε\| (V) | \|ε\|/N (V) |
|---|---|---|---|
| 1 | 50 | 0.2034 | 0.004068 |
| 2 | 100 | 0.4068 | 0.004068 |
| 3 | 200 | 0.8135 | 0.004068 |
| 4 | 400 | 1.6270 | 0.004068 |
| 5 | 1000 | 4.0676 | 0.004068 |

**Result.** ε ∝ N. **Reason.** Each turn links the same flux; the flux linkage NΦ scales with N.
*(In a real coil more turns also means more wire resistance, so the current would not grow as
much; here R is held fixed on purpose.)*

---

## Experiment C — effect of magnet speed

**Aim.** Study how the peak EMF depends on the speed. **Held constant.** B₀ (field distribution),
N, A, θ, path, R.

1. **Virtual experiments → C · Magnet speed**; values `0.25, 0.5, 1, 2, 3, 4, 5`.
2. Press **Run experiment C**.

| S.No. | v (m/s) | Peak \|ε\| (V) | \|ε\|/v (V·s/m) |
|---|---|---|---|
| 1 | 0.25 | 0.2034 | 0.8135 |
| 2 | 0.5 | 0.4068 | 0.8135 |
| 3 | 1 | 0.8135 | 0.8135 |
| 4 | 2 | 1.6270 | 0.8135 |
| 5 | 5 | 4.0676 | 0.8135 |

**Result.** Peak ε ∝ v. **Reason.** dΦ/dt = (dΦ/dx)·v: the same flux change happens in less time.

**Extra.** Set *Approach* motion, run with v = 0.5 and v = 2 m/s, and compare **Net charge** in
*Results and conclusion*: it is the same (−NΔΦ/R ≈ −1.92 mC) — speed changes the peak, not the charge.

---

## Experiment D — Lenz's law

**Aim.** Verify that the induced current opposes the change of flux.

1. **Virtual experiments → D · Lenz's law** → **Run experiment D**.

| Case | Flux along n̂ | Dominant ε (V) | Current (seen from magnet) | Coil face toward magnet | Effect |
|---|---|---|---|---|---|
| N pole approaches | increasing | −0.695 | anticlockwise | N | repels, opposing the approach |
| N pole moves away | decreasing | +0.695 | clockwise | S | attracts, opposing the separation |
| S pole approaches | decreasing | +0.695 | clockwise | S | repels, opposing the approach |
| S pole moves away | increasing | −0.695 | anticlockwise | N | attracts, opposing the separation |

(Approach/move-away between 30 cm and 3 cm.)

2. Note the two extra demonstrations shown under the table:
   * **Coil edge-on (θ = 90°)**: the magnet moves exactly as before but the flux stays zero, so
     there is no EMF and nothing is opposed.
   * **Uniform field rising at 0.5 T/s, nothing moving**: a steady EMF ε = −NA dB/dt appears
     (−0.5 V for N = 200, A = 0.005 m²). The current opposes the growing field although there is
     no motion.

**Conclusion.** The induced current opposes the change in flux (not motion as such) — Lenz's law.

---

## Things to try

* **Stationary magnet:** Motion → *Stationary*. Flux is non-zero but constant; EMF = 0.
* **Tilt:** θ = 0°, 45°, 90°, 135°, 180°. The EMF shrinks toward 90° and reverses beyond it.
* **Resistance:** change R; the EMF is unchanged, the current scales as 1/R.
* **Closest distance:** in *Approach* mode, 3 cm vs 6 cm — the 1/r³ fall-off makes a big difference.
* **Average vs peak:** in *Pass through* mode the average EMF is zero but the peak is not.

---

## Viva questions (with short answers)

1. *State Faraday's law.* The induced EMF equals the negative rate of change of flux linkage, ε = −N dΦ/dt.
2. *What does the minus sign mean?* Lenz's law: the induced current opposes the change of flux.
3. *Why is the EMF zero when the magnet is at the centre of the coil?* The flux is at its maximum
   there, so dΦ/dt = 0 for an instant.
4. *Why does the EMF change sign during pass-through?* The flux first increases, then decreases.
5. *Why is the EMF proportional to speed?* For a fixed path dΦ/dt = (dΦ/dx)·v.
6. *Does a strong stationary magnet induce an EMF?* No; only a changing flux does.
7. *Why is no EMF induced when the coil is edge-on?* No field lines pass through its area, so the
   flux stays zero even though the magnet moves.
8. *What is the difference between peak and average EMF?* Peak = largest instantaneous value;
   average = −NΔΦ/Δt over the motion, which is zero when the flux returns to its starting value.
9. *Where does the energy come from?* From the work done against the opposing magnetic force on
   the moving magnet (Lenz's law ensures energy conservation).
10. *What are the model's limitations?* Point-dipole magnet, thin coil, no self-inductance or
    force on the magnet, instantaneous start/stop.
11. *How is dΦ/dt calculated?* Central finite differences on the sampled flux, checked against the
    analytic derivative.
12. *How was the flux model checked?* Against direct numerical integration of **B**·d**A**
    (agreement ~10⁻¹⁵) and the closed-form on-axis formula.
