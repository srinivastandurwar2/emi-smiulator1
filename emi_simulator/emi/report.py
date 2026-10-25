"""Self-contained HTML experiment report (images embedded, works offline).

Open the downloaded file in any browser; use the browser's Print → Save as PDF
to obtain a PDF for submission.
"""
from __future__ import annotations

import base64
import datetime as _dt
import html

from . import plots_mpl as pm
from .analysis import SWEEP_INFO, fmt_si, params_table, sweep_conclusion

CSS = """
body{font-family:Georgia,'Times New Roman',serif;color:#1d2733;max-width:900px;margin:32px auto;
     padding:0 20px;line-height:1.55}
h1{font-size:1.7em;margin-bottom:0}h2{font-size:1.25em;border-bottom:1px solid #c9d2dc;padding-bottom:4px;
     margin-top:2em}h3{font-size:1.05em}
table{border-collapse:collapse;margin:10px 0;font-family:Arial,Helvetica,sans-serif;font-size:.85em}
td,th{border:1px solid #c9d2dc;padding:4px 8px;text-align:left}th{background:#eef2f6}
.eq{font-family:'Cambria Math','STIX Two Math',Georgia,serif;font-size:1.1em;background:#f4f6f8;
    padding:8px 12px;border-left:3px solid #0f8b6e;margin:8px 0}
img{max-width:100%;border:1px solid #e3e8ee;margin:8px 0}.pass{color:#0f7a3a;font-weight:bold}
.fail{color:#b3261e;font-weight:bold}.muted{color:#5c6b7a;font-size:.9em}
"""


def _img(fig) -> str:
    data = base64.b64encode(pm.to_png_bytes(fig, dpi=130)).decode()
    return f'<img src="data:image/png;base64,{data}" alt="graph">'


def _table(rows: list[dict], max_rows: int = 60) -> str:
    if not rows:
        return "<p class='muted'>No rows.</p>"
    cols = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = []
    for r in rows[:max_rows]:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"<td>{v:.5g}</td>" if isinstance(v, float) else f"<td>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def build_report(res, title: str = "Electromagnetic induction — virtual experiment report",
                 student: str = "", sweeps: dict | None = None, lenz: dict | None = None,
                 observations: list[dict] | None = None, validation: list[dict] | None = None) -> str:
    p = res.params
    m = res.metrics()
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style></head><body>"]
    parts.append(f"<h1>{html.escape(title)}</h1>")
    meta = _dt.datetime.now().strftime("%d %B %Y, %H:%M")
    parts.append(f"<p class='muted'>{html.escape(student) + ' — ' if student else ''}Generated {meta} "
                 f"by the local EMI simulator. All values are computed by the model, not measured.</p>")

    parts.append("<h2>1. Aim</h2><p>To study how the EMF induced in a coil depends on the magnet's "
                 "strength, the number of turns and the speed of the magnet, and to verify Faraday's "
                 "and Lenz's laws using a numerical model.</p>")

    parts.append("<h2>2. Theory</h2>")
    parts.append("<div class='eq'>ε = −N dΦ<sub>B</sub>/dt &nbsp;&nbsp;&nbsp; (Faraday's law, with Lenz's minus sign)</div>")
    parts.append("<div class='eq'>Φ<sub>B</sub> = ∫ <b>B</b>·d<b>A</b> &nbsp; → &nbsp; Φ<sub>B</sub> = B A cos θ "
                 "for a uniform field</div>")
    parts.append("<div class='eq'>Dipole on the coil axis: Φ(x) = μ<sub>0</sub> m a² / [2 (a² + x²)<sup>3/2</sup>]"
                 "</div>")
    parts.append("<div class='eq'>I(t) = ε(t) / R</div>")
    parts.append("<p>The magnet is a point magnetic dipole of moment m = 2π r<sub>ref</sub>³ B₀ / μ₀ "
                 "(r<sub>ref</sub> = 2 cm), so B₀ is the field on the magnet's axis 2 cm from its centre. "
                 "The flux through each circular turn (radius a = √(A/π)) is computed exactly for any "
                 "position and tilt using the reciprocity theorem and elliptic integrals; dΦ/dt is "
                 "obtained with second-order central differences.</p>")

    parts.append("<h2>3. Parameters</h2><table>" + "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in params_table(p)) + "</table>")

    parts.append("<h2>4. Results of the selected run</h2>")
    rows = [("Peak positive EMF", fmt_si(m["peak_pos_emf_V"], "V")),
            ("Peak negative EMF", fmt_si(m["peak_neg_emf_V"], "V")),
            ("Maximum |EMF|", fmt_si(m["max_abs_emf_V"], "V")),
            ("Average EMF over the motion (−NΔΦ/Δt)", fmt_si(m["average_emf_V"], "V")),
            ("Maximum |current|", fmt_si(m["max_abs_current_A"], "A")),
            ("Maximum |flux| per turn", fmt_si(m["max_abs_flux_Wb"], "Wb")),
            ("Change in flux per turn (final − initial)", fmt_si(m["delta_flux_Wb"], "Wb")),
            ("Net charge through the circuit", fmt_si(m["net_charge_C"], "C")),
            ("Dipole moment of the magnet", f"{m['dipole_moment_Am2']:.4g} A·m²")]
    parts.append("<table>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>")
    parts.append(_img(pm.scene_snapshot(res)))
    parts.append(_img(pm.overview_figure(res)))

    if sweeps:
        parts.append("<h2>5. Parameter experiments</h2>")
        for name, sw in sweeps.items():
            info = SWEEP_INFO.get(name, {"label": name})
            obs, theory = sweep_conclusion(sw)
            parts.append(f"<h3>Peak EMF vs {html.escape(info['label'])}</h3>")
            parts.append(_img(pm.sweep_plot(sw)))
            parts.append(f"<p>{html.escape(obs)}</p><p>{html.escape(theory)}</p>")

    if lenz:
        parts.append("<h2>6. Lenz's law</h2>")
        parts.append(_img(pm.lenz_plot(lenz)))
        parts.append(_table(lenz["cases"]))
        parts.append("<p>In every case the induced current makes the coil face nearest the magnet "
                     "repel an approaching pole and attract a receding one. With the coil edge-on "
                     "(θ = 90°) the magnet still moves but the flux does not change, so no current "
                     "flows: the induced current opposes the change of flux, not motion itself.</p>")

    if observations:
        parts.append("<h2>7. Observation table</h2>")
        parts.append(_table(observations))

    if validation:
        parts.append("<h2>8. Model validation</h2><table><tr><th>Check</th><th>Expected</th>"
                     "<th>Measured</th><th>Result</th></tr>")
        for c in validation:
            cls, word = ("pass", "Pass") if c["Passed"] else ("fail", "Fail")
            parts.append(f"<tr><td>{html.escape(c['Check'])}</td><td>{html.escape(c['Expected'])}</td>"
                         f"<td>{html.escape(c['Measured'])}</td><td class='{cls}'>{word}</td></tr>")
        parts.append("</table>")

    parts.append("<h2>Conclusion</h2><p>The induced EMF is proportional to the rate of change of "
                 "flux linkage. For the same magnet path, the peak EMF is directly proportional to "
                 "the magnet strength, the number of turns and the speed. A stationary magnet produces "
                 "no EMF, and the EMF reverses sign when the motion or the polarity is reversed. The "
                 "direction of the induced current always opposes the change of flux, as stated by "
                 "Lenz's law.</p>")
    parts.append("<h2>Assumptions and limitations</h2><ul>"
                 "<li>Magnet = ideal point dipole; accurate when the magnet is small compared with its "
                 "distance from the coil, approximate when it is inside or very close to the coil.</li>"
                 "<li>Thin coil: all N turns link the same flux.</li>"
                 "<li>Purely resistive circuit (I = ε/R); coil self-inductance and the magnetic force on "
                 "the magnet are neglected, so the magnet's speed stays constant.</li>"
                 "<li>Instantaneous start and stop of the motion (constant-speed segments).</li></ul>")
    parts.append("</body></html>")
    return "".join(parts)
