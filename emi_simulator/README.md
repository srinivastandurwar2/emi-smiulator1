# Electromagnetic Induction Virtual Lab

An interactive, locally hosted simulator of electromagnetic induction for a B.Tech first-year
Engineering Physics project. A bar magnet moves along the axis of a coil; the program calculates
the magnetic flux through the coil, the induced EMF (Faraday's law) and the induced current and
its direction (Lenz's law) from first principles, and shows them in an animated virtual lab with a
voltmeter, a centre-zero galvanometer and live graphs.

Everything runs on your own computer. After the one-time installation no internet connection is
needed.

---

## 1. What you can do

* Change the magnet strength **B₀** (0.01–1 T), turns **N** (10–1000), speed **v** (0.01–5 m/s),
  coil area **A** (0.001–0.1 m²), coil tilt **θ** (0–180°), start and closest distance, motion
  (pass through, push in and pull out, approach, move away, stationary), which pole faces the coil,
  resistance **R** and the simulated duration.
* Watch the animation: magnet, field lines computed from the model, coil with current symbols
  (⊙ ⊗), induced poles on the coil faces, a galvanometer needle, an EMF/current/flux readout and a
  "seen from the magnet" view of the current's direction.
* Read six live graphs: EMF vs time, peak EMF vs B₀, vs N, vs v, flux (with dΦ/dt) vs time,
  current vs time.
* Run four guided experiments (A: magnet strength, B: turns, C: speed, D: Lenz's law). Results go
  into an observation table that you can export as CSV.
* Download every graph as PNG and a full experiment report (HTML; print to PDF from the browser).
* See 14 automatic validation checks against known theoretical results.

## 2. Folder structure

```
emi_simulator/
├── app.py                     Streamlit application (the user interface)
├── emi/                       Simulation package
│   ├── params.py              Parameters, ranges, defaults, input validation
│   ├── physics.py             Dipole + loop-field flux model, trajectory, Faraday/Lenz, I = ε/R
│   ├── analysis.py            Parameter sweeps, fits, Lenz cases, observation rows, formatting
│   ├── validation.py          14 checks against theory
│   ├── explain.py             Plain-language explanations that react to parameter changes
│   ├── scene.py               Geometry of the side view (exact dipole field lines, coil, magnet)
│   ├── plots_plotly.py        Animated lab and interactive graphs (Plotly)
│   ├── plots_mpl.py           Static graphs for PNG export and the report (Matplotlib)
│   └── report.py              Self-contained HTML experiment report
├── tests/test_physics.py      Automated tests (54 cases)
├── scripts/
│   ├── generate_examples.py   Re-creates everything in examples/ from the model
│   └── run_tests.py           Runs the tests (with or without pytest)
├── examples/                  Example graphs, CSV observation tables and a sample report
├── docs/
│   ├── THEORY.md              Equations, model, numerical method, assumptions
│   └── EXPERIMENT_GUIDE.md    Step-by-step experiment instructions and viva questions
├── .streamlit/config.toml     Local server settings and theme
├── requirements.txt
├── run_app.bat                Double-click launcher (Windows)
└── run_app.sh                 Launcher (macOS / Linux)
```

## 3. Installation (one time, needs internet)

You need **Python 3.11 or newer**. Check with `python --version` (Windows) or
`python3 --version` (macOS/Linux). Download Python from python.org if needed; on Windows tick
**"Add python.exe to PATH"** in the installer.

Open a terminal in the `emi_simulator` folder and run:

**Windows (Command Prompt or PowerShell)**
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

The virtual environment (`.venv`) is optional but keeps the project's packages separate from
anything else on your computer.

### Installing on a computer that has no internet (e.g. a lab PC)
On a computer with internet and the **same operating system and Python version**:
```bash
pip download -r requirements.txt -d wheels
```
Copy the project folder together with `wheels/` to the lab PC, then:
```bash
pip install --no-index --find-links wheels -r requirements.txt
```

## 4. Running the application

```bash
streamlit run app.py
```
(or `python -m streamlit run app.py`, or double-click `run_app.bat` on Windows /
run `./run_app.sh` on macOS/Linux; activate the virtual environment first if you made one).

Your browser opens **http://localhost:8501**. If it doesn't, open that address yourself. The app
is served only to your own computer (`localhost`). Stop it with **Ctrl + C** in the terminal.

## 5. Using the app

1. **Sidebar — Parameter controls.** Every change re-runs the simulation instantly; the animation
   restarts with the new values, and the meters, graphs and explanation panel update.
   **Reset to defaults** restores the starting values.
2. **Interactive simulation.** The animation plays automatically. Use ▶ Play, ❚❚ Pause, ⟲ Replay and
   the time slider. Below it: peak/average EMF, current and flux values, an explanation of what is
   happening, what changed since your last setting, and an "inspect one instant" slider.
3. **Live graphs.** Hover for exact values; the camera icon saves a graph; **Prepare all graphs as
   PNG** downloads them all.
4. **Virtual experiments.** Enter the values to test (comma separated) and press **Run experiment**.
   Each run is added to the observation table, with an automatic observation and comparison with
   theory.
5. **Observation table.** Export as CSV (opens in Excel/LibreOffice).
6. **Results and conclusion.** Results of the current run, time-series CSV, validation checks and
   **Build report** → **Download report (HTML)**. Open it in a browser and use *Print → Save as
   PDF* for a PDF.

Default settings: B₀ = 0.3 T, N = 200, v = 1 m/s, A = 0.005 m², θ = 0°, magnet passes through
the coil from 30 cm away, N pole facing, R = 10 Ω. This gives a peak EMF of about ±0.81 V.

## 6. Physics model in brief

Full details: [docs/THEORY.md](docs/THEORY.md).

* **Faraday:** ε(t) = −N dΦ/dt, computed with central finite differences; **I(t) = ε(t)/R**.
* **Magnet:** a point magnetic dipole. **B₀** is the field on the magnet's own axis 2 cm from its
  centre, which fixes the dipole moment m = 2π(0.02 m)³B₀/μ₀. Changing B₀ scales the field
  everywhere; the shape of the field and the coil, path and speed stay the same.
* **Flux:** for any magnet position and coil tilt, Φ = **m**·**B**_loop/I (reciprocity theorem),
  where **B**_loop is the exact field of a circular loop (complete elliptic integrals). On the axis
  this is Φ = μ₀ m a² / [2(a² + x²)^{3/2}]. This was checked against direct numerical integration
  of **B**·d**A** over the coil (agreement to ~10⁻¹⁵).
* **Uniform-field case** (Φ = BA cos θ, ε = −NA cos θ dB/dt) is used as a validation case and in
  experiment D.
* **Sign convention:** positive flux is along the coil's area vector n̂; positive EMF/current
  circulate around n̂ by the right-hand rule. At θ = 0 a positive current looks clockwise when
  viewed from the magnet's starting side.

**Assumptions and limitations:** point-dipole magnet (approximate when the magnet is inside or
very close to the coil; the drawn magnet size is only illustrative); thin coil (all turns link the
same flux); resistive circuit only (no self-inductance, no force on the magnet, so speed stays
constant); instantaneous start/stop of motion.

## 7. Tests and example outputs

```bash
python scripts/run_tests.py          # or: python -m pytest -q
python scripts/generate_examples.py  # regenerates examples/ (graphs, CSVs, sample_report.html)
```
`examples/EXAMPLE_RESULTS.md` summarises the example observation tables and validation results.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `'streamlit' is not recognized` / `command not found` | Activate the virtual environment, or use `python -m streamlit run app.py`. |
| `'python' is not recognized` (Windows) | Reinstall Python with "Add python.exe to PATH", or use the `py` launcher: `py -m venv .venv`. |
| PowerShell refuses to run `activate` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt. |
| `ModuleNotFoundError: No module named 'emi'` | Start the app from inside the `emi_simulator` folder (where `app.py` is). |
| `ModuleNotFoundError` for numpy/plotly/… | `pip install -r requirements.txt` inside the activated environment. |
| Terminal asks for an e-mail address on first run | Press Enter to skip; it is Streamlit's optional welcome prompt. |
| Browser doesn't open | Open http://localhost:8501 manually. |
| `Port 8501 is already in use` | Close the other Streamlit window, or run `streamlit run app.py --server.port 8502`. |
| Animation area is blank | Sidebar → Display → Animation player → "Standard chart (press Play)". Make sure the browser is up to date (Chrome, Edge, Firefox). |
| App feels slow on an old PC | Use the "Standard chart" player; keep the browser zoom at 100 %. |
| Python version too old | Install Python 3.11+ and recreate the virtual environment. |
| pip install fails behind a college proxy | `pip install --proxy http://user:pass@proxy:port -r requirements.txt`, or use the offline wheel method above. |
| Want a dark interface | ⋮ menu (top right) → Settings → Theme; set Graph theme to Dark in the sidebar. |
