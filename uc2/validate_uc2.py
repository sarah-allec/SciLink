"""Compare computed vs measured UC2 properties and report the trend.

Reads each member's ``results.json`` (written by ``run_uc2.py --stage analyze``)
and lines the computed density / viscosity / water T1 up against the measured
references. Density and viscosity are the *validation* observables: if the model
reproduces them, the T1 *prediction* is trustworthy; if it doesn't, the report
flags which observable is off and in which direction — the force-field limitation
the use case exists to surface.

Importable (``report(comps, runs_dir)``) or standalone:
    python validate_uc2.py [--runs-dir uc2/runs]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import references  # noqa: E402


# Fuzzy match a property name to a physical quantity, since the exact key comes
# from whichever skill computed it.
_MATCHERS = {
    "density": lambda k: "densit" in k,
    "viscosity": lambda k: "viscos" in k,
    "water_T1": lambda k: "t1" in k or "relax" in k,
}


def _extract(results: dict) -> dict:
    """Pull a scalar value per quantity from a SimulationAnalysisAgent result."""
    out = {}
    props = (results or {}).get("results", {}) or {}
    for quantity, match in _MATCHERS.items():
        for key, entry in props.items():
            if not match(key.lower()):
                continue
            val = entry.get("value") if isinstance(entry, dict) else entry
            verified = None
            if isinstance(entry, dict):
                v = entry.get("verification")
                if isinstance(v, dict):
                    verified = v.get("plausible")
            out[quantity] = {"value": val, "verified": verified, "key": key}
            break
    return out


def _pct(computed, measured):
    if computed is None or measured in (None, 0):
        return None
    return 100.0 * (computed - measured) / measured


def _direction(series):
    """'increases' / 'decreases' / 'flat' / None for a value sequence ordered
    by rising EIS fraction."""
    vals = [v for v in series if isinstance(v, (int, float))]
    if len(vals) < 2:
        return None
    d = vals[-1] - vals[0]
    if abs(d) < 1e-9:
        return "flat"
    return "increases" if d > 0 else "decreases"


def report(comps, runs_dir: Path):
    runs_dir = Path(runs_dir)
    rows = []
    for comp in comps:
        rj = runs_dir / comp["label"] / "results.json"
        computed = _extract(json.loads(rj.read_text())) if rj.exists() else {}
        rows.append((comp, computed))

    def line(label, cval, mval, unit, verified=None):
        c = f"{cval:.4g}" if isinstance(cval, (int, float)) else "--"
        m = f"{mval:.4g}" if isinstance(mval, (int, float)) else "--"
        pct = _pct(cval if isinstance(cval, (int, float)) else None, mval)
        p = f"{pct:+.1f}%" if pct is not None else "  -- "
        vflag = "" if verified is None else ("  [verified]" if verified
                                             else "  [UNVERIFIED]")
        print(f"    {label:12s} computed {c:>10s}  measured {m:>10s} "
              f"{unit:8s} {p:>8s}{vflag}")

    print("\n=== UC2 validation: computed vs measured (298.15 K) ===\n")
    for comp, computed in rows:
        print(f"  {comp['label']}  (H2O:EIS {comp['water_ratio']}:{comp['eis_ratio']})")
        d = computed.get("density", {})
        v = computed.get("viscosity", {})
        t = computed.get("water_T1", {})
        line("density", d.get("value"), comp["density_g_cm3"], "g/cm^3",
             d.get("verified"))
        line("viscosity", v.get("value"), comp["viscosity_mPa_s"], "mPa*s",
             v.get("verified"))
        line("water T1", t.get("value"), comp["water_T1_s"], "s",
             t.get("verified"))
        if comp["notes"]:
            print(f"       note: {comp['notes']}")
        print()

    # Trend check — the crux of the use case.
    print("=== trend across the series (rising EIS fraction) ===\n")
    for quantity, meas_key in (("density", "density_g_cm3"),
                               ("viscosity", "viscosity_mPa_s"),
                               ("water_T1", "water_T1_s")):
        comp_series = [c.get(quantity, {}).get("value") for _, c in rows]
        meas_series = [comp[meas_key] for comp, _ in rows]
        cd = _direction(comp_series)
        md = _direction(meas_series)
        expected = references.MEASURED_TREND.get(
            "water_T1" if quantity == "water_T1" else quantity, "")
        agree = (cd is not None and md is not None
                 and cd == md)
        verdict = ("OK" if agree else
                   "MISMATCH" if (cd and md) else "insufficient data")
        print(f"  {quantity:10s} computed: {cd or '--':10s} "
              f"measured: {md or '--':10s} -> {verdict}")
        if quantity in ("density", "viscosity") and cd and md and not agree:
            print(f"      ^ validation observable trend is wrong "
                  f"(measured {expected}); the prediction downstream is not "
                  f"trustworthy until the force field is corrected.")
    print()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    ap.add_argument("--members", default="all")
    args = ap.parse_args()
    if args.members == "all":
        comps = references.COMPOSITIONS
    else:
        want = {m.strip() for m in args.members.split(",")}
        comps = [references.by_label(l) for l in references.labels() if l in want]
    report(comps, Path(args.runs_dir))


if __name__ == "__main__":
    main()
