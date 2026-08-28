"""Check whether the viscosity TREND across the composition series is resolved.

Reads each composition's replica results (``runs_dir/<member>/rep_*/results.json``,
one per seed), takes the mean viscosity over its replicas, and asks SciLink's
``TrendCritic`` whether the trend DIRECTION is now clear beyond the replica
spread. Prints the verdict and, if it is not clear yet, roughly how many more
seeds to run.

For a hard Green-Kubo case the trustworthy signal is the trend DIRECTION, not the
absolute value — so this checks the direction, and treats overlapping/noisy
points as "not resolved yet, run more" rather than a failure.

Importable (``report(comps, runs_dir)``) or standalone:
    python converge_uc2.py [--runs-dir ...] [--members 80-20,70-30,60-40]
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import references  # noqa: E402

MIN_REPLICAS = 3        # do not judge a trend on fewer than this per composition


def _replica_viscosities(member_dir: Path) -> list:
    """Every replica's viscosity for one composition (rep_*/results.json)."""
    vals = []
    for rep in sorted(member_dir.glob("rep_*")):
        rj = rep / "results.json"
        if not rj.exists():
            continue
        try:
            res = (json.loads(rj.read_text()).get("results") or {})
        except Exception:
            continue
        for key, entry in res.items():
            if "viscos" not in key.lower():
                continue
            v = entry.get("value") if isinstance(entry, dict) else entry
            if isinstance(v, (int, float)):
                vals.append(float(v))
            break
    return vals


def report(comps, runs_dir):
    runs_dir = Path(runs_dir)
    rows, series, counts = [], [], []
    for comp in comps:
        vals = _replica_viscosities(runs_dir / comp["label"])
        n = len(vals)
        counts.append(n)
        mean = statistics.fmean(vals) if vals else None
        sem = (statistics.stdev(vals) / (n ** 0.5)) if n >= 2 else None
        rows.append((comp, n, mean, sem))
        if mean is not None:
            series.append({"point": f'{comp["water_ratio"]}:{comp["eis_ratio"]}',
                           "value": mean})

    print("\n=== viscosity replicas per composition ===\n")
    for comp, n, mean, sem in rows:
        m = f"{mean:.3g}" if mean is not None else "--"
        s = f" ± {sem:.2g}" if sem else ""
        print(f"  {comp['label']}: {n} replicas, mean {m}{s} mPa*s")

    ready = [r for r in rows if r[2] is not None]
    if len(ready) < 2:
        print("\nNot enough compositions have replicas yet — run the batch first.")
        return
    thin = [r[0]['label'] for r in rows if 0 < r[1] < MIN_REPLICAS]
    if thin:
        print(f"\nNote: fewer than {MIN_REPLICAS} replicas for {', '.join(thin)} "
              "— the trend check is provisional until they catch up.")

    try:
        from scilink.agents.sim_agents.critics import TrendCritic
        critic = TrendCritic(
            api_key=os.environ.get("SCILINK_API_KEY"),
            base_url=os.environ.get("SCILINK_BASE_URL"),
            model_name=os.environ.get("SCILINK_MODEL", "claude-opus-4-8-project"))
    except Exception as e:
        print(f"\n(TrendCritic unavailable: {e})")
        return

    # Give the critic the replica spread so it separates a real trend from noise.
    spread = "; ".join(
        f"{comp['label']} = {mean:.3g}" + (f"±{sem:.2g}" if sem else " (1 replica)")
        for comp, n, mean, sem in rows if mean is not None)
    reference_context = (
        "Measured viscosity INCREASES with sulfone fraction (sulfones are more "
        "viscous than water). Each plotted point is a mean over independent MD "
        f"replicas; the per-composition replica spread is: {spread}. Judge the "
        "trend resolved only if its direction is clear beyond this replica "
        "spread; overlapping points mean it is not resolved yet.")
    verdict = critic.assess(
        series, quantity="shear viscosity", units=" mPa*s",
        parameter="sulfone (EIS) fraction",
        system_description=("1 M Zn(OTf)2 in H2O / ethyl-isopropyl-sulfone, "
                            "composition series"),
        reference_context=reference_context)

    print("\n=== SciLink trend check (TrendCritic) ===\n")
    print(f"  expected direction: {verdict.get('expected_direction')}")
    print(f"  observed direction: {verdict.get('observed_direction')}")
    print(f"  consistent with experiment: {verdict.get('consistent')}"
          f"   verdict: {verdict.get('verdict')}")
    if verdict.get("reasoning"):
        print(f"  reasoning: {verdict['reasoning']}")

    observed = str(verdict.get("observed_direction") or "").lower()
    resolved = observed in ("increasing", "decreasing")
    print()
    if resolved:
        if verdict.get("consistent"):
            print("  -> Trend is clear and MATCHES experiment. Done sampling.")
        else:
            print("  -> Trend is clear but CONTRADICTS experiment — a real "
                  "method failure. Done sampling; this is the escalation signal.")
    else:
        need = max(MIN_REPLICAS - min(counts), 4) if counts else 4
        print(f"  -> Trend not clear yet (points overlap). Run about {need} more "
              "seeds per composition, then check again.")


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
