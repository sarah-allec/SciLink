#!/usr/bin/env python
"""UC2 driver — 1 M Zn(OTf)2 in H2O/EIS, the composition series.

Drives the simulate pipeline over the four solvent ratios (80-20 -> 50-50) with
a single high-level request per member; the agents resolve chemistry, box, force
field, and the staged MD deck. `derive_observables` turns the goal into the
observable-requirements contract (dense stress sampling for Green-Kubo viscosity,
sub-picosecond dumps for 1H T1), which is threaded into generation and the
pre-run coverage gate.

Stages
  preview   generate + validate decks only, autopilot (human-in-the-loop, login
            node). No MD is run. Review the box, FF, observable contract, and the
            dry-run gate before committing GPU time.
  full      generate + run + refine through the executor, autonomous (compute
            node, driven by submit_uc2.sbatch).
  analyze   compute density / viscosity / T1 from each member's output via the
            SimulationAnalysisAgent (writes results.json per member).
  validate  compare computed vs measured references and report the trend.

Environment (all optional except an API key)
  SCILINK_MODEL      model name        (default: claude-opus-4-8-project)
  SCILINK_BASE_URL   OpenAI-compatible proxy URL; when set, auth uses
                     SCILINK_API_KEY, else the direct path uses ANTHROPIC_API_KEY
  FUTUREHOUSE_API_KEY  optional, for literature-grounded steps
  LAMMPS_CMD         run-command template, must contain {script}
                     (default: "lmp -in {script}")
  UC2_STAGE_TIMEOUT  per-stage wall-clock seconds (default: 43200)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import references  # noqa: E402  (uc2/references.py)


def _request(comp: dict) -> str:
    """High-level, goal-first request for one composition — no pre-chewed
    SMILES / molecule counts / box: the structure and force-field agents resolve
    those. Naming the target observables lets the deriver build the sampling
    contract."""
    w, e = comp["water_ratio"], comp["eis_ratio"]
    return (
        f"Build and equilibrate a {references.SALT_CONCENTRATION_M:g} M "
        f"{references.SALT} electrolyte in a mixed solvent of water and ethyl "
        f"isopropyl sulfone at a {w}:{e} water-to-sulfone volume ratio, at "
        f"{references.TEMPERATURE_K:g} K and {references.PRESSURE_ATM:g} atm. "
        "Use a box large enough for reliable liquid-state properties. Run staged "
        "molecular dynamics (energy minimization, then NPT equilibration, then a "
        "production run) long enough to compute the mass density, the shear "
        "viscosity by the Green-Kubo method, and the 1H spin-lattice (T1) "
        "relaxation time of the water protons. Sample the pressure/stress tensor "
        "densely enough for the Green-Kubo integral to converge, and dump atomic "
        "trajectories at sub-picosecond intervals so reorientational dynamics are "
        "resolved."
    )


def _credentials():
    base_url = os.environ.get("SCILINK_BASE_URL") or None
    model = os.environ.get("SCILINK_MODEL", "claude-opus-4-8-project")
    if base_url:
        api_key = os.environ.get("SCILINK_API_KEY")
        if not api_key:
            sys.exit("SCILINK_BASE_URL is set but SCILINK_API_KEY is not "
                     "(the proxy needs it).")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("no SCILINK_BASE_URL -> direct path needs ANTHROPIC_API_KEY.")
    return model, base_url, api_key


def _run_member(comp: dict, stage: str, runs_dir: Path) -> dict:
    from scilink.agents.sim_agents import simulation_pipeline as sp

    model, base_url, api_key = _credentials()
    out_dir = runs_dir / comp["label"]
    out_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        scale="molecular_dynamics", software="lammps",
        structure_class="condensed",  # many-molecule solution box
        output_dir=str(out_dir),
        api_key=api_key, base_url=base_url, model_name=model,
        futurehouse_api_key=os.environ.get("FUTUREHOUSE_API_KEY"),
        derive_observables=True,
        staged=True,
        validate=True,
    )

    if stage == "preview":
        # Human-in-the-loop generation only; nothing executes.
        result = sp.run_complete_workflow(
            _request(comp), autonomy="autopilot",
            executor=None, run_command=None, **common)
    else:  # full
        from scilink.agents.sim_agents.refinement import LocalExecutor
        timeout = int(os.environ.get("UC2_STAGE_TIMEOUT", "43200"))
        run_cmd = os.environ.get("LAMMPS_CMD", "lmp -in {script}")
        result = sp.run_complete_workflow(
            _request(comp), autonomy="autonomous",
            executor=LocalExecutor(timeout=timeout), run_command=run_cmd,
            max_run_cycles=3, **common)

    (out_dir / f"result_{stage}.json").write_text(json.dumps(result, indent=2,
                                                             default=str))
    status = result.get("final_status", "unknown")
    print(f"[{comp['label']}] {stage} -> {status}")
    return result


def _analyze_member(comp: dict, runs_dir: Path) -> dict:
    from scilink.agents.sim_agents import SimulationAnalysisAgent

    model, base_url, api_key = _credentials()
    out_dir = runs_dir / comp["label"]
    if not out_dir.exists():
        print(f"[{comp['label']}] no run dir; skipping analyze")
        return {}

    agent = SimulationAnalysisAgent(
        api_key=api_key, base_url=base_url, model_name=model,
        output_dir=str(out_dir / "analysis"))
    goal = ("Compute the mass density (g/cm^3), the shear viscosity (mPa*s) by "
            "Green-Kubo, and the 1H spin-lattice T1 (s) of the water protons "
            "from this finished electrolyte MD run.")
    res = agent.run_analysis(goal, run_dir=str(out_dir))
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"[{comp['label']}] analyze -> {res.get('status')} "
          f"({', '.join(res.get('results', {}) or {}) or 'no properties'})")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["preview", "full", "analyze", "validate"])
    ap.add_argument("--members", default="all",
                    help="comma-separated labels (e.g. 80-20,70-30) or 'all'")
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if args.members == "all":
        comps = references.COMPOSITIONS
    else:
        want = {m.strip() for m in args.members.split(",")}
        comps = [references.by_label(l) for l in references.labels() if l in want]
    if not comps:
        sys.exit(f"no members matched {args.members!r}")

    if args.stage == "validate":
        import validate_uc2
        validate_uc2.report(comps, runs_dir)
        return

    for comp in comps:
        if args.stage in ("preview", "full"):
            _run_member(comp, args.stage, runs_dir)
        elif args.stage == "analyze":
            _analyze_member(comp, runs_dir)


if __name__ == "__main__":
    main()
