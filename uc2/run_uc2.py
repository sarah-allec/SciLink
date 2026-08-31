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

# Human-in-the-loop knob. The agent's own duration choice (~10 ns) is optimistic
# for a Green-Kubo viscosity on a viscous electrolyte — the stress autocorrelation
# converges slowly, so a short run gives a noisy integral. Until the convergence
# loop is closed automatically (see issue: auto-extend production on a non-
# converged post-run observable), a domain scientist sets the production length
# here. The post-run analysis still reports whether the integral actually
# plateaued, so this choice is confirmed rather than assumed.
PRODUCTION_NS = float(os.environ.get("UC2_PRODUCTION_NS", "25"))


def _request(comp: dict) -> str:
    """High-level, goal-first request for one composition — no pre-chewed
    SMILES / molecule counts / box: the structure and force-field agents resolve
    those. Naming the target observables lets the deriver build the sampling
    contract; the production length is set explicitly (see PRODUCTION_NS)."""
    w, e = comp["water_ratio"], comp["eis_ratio"]
    return (
        f"Build and equilibrate a {references.SALT_CONCENTRATION_M:g} M "
        f"{references.SALT} electrolyte in a mixed solvent of water and ethyl "
        f"isopropyl sulfone at a {w}:{e} water-to-sulfone volume ratio, at "
        f"{references.TEMPERATURE_K:g} K and {references.PRESSURE_ATM:g} atm. "
        "Use a box large enough for reliable liquid-state properties. Run staged "
        "molecular dynamics: energy minimization, then NPT equilibration to "
        "converge the density, then a production (NVT) run of at least "
        f"{PRODUCTION_NS:g} ns. The production must be long enough for the "
        "Green-Kubo shear-viscosity integral to converge — this is a viscous "
        "electrolyte, so the stress autocorrelation converges slowly and a short "
        "run gives a noisy viscosity. Compute the mass density, the shear "
        "viscosity by the Green-Kubo method, and the 1H spin-lattice (T1) "
        "relaxation time of the water protons. Sample the pressure/stress tensor "
        "densely (every few fs) so the Green-Kubo integral is well resolved, and "
        "dump the water-hydrogen trajectory at sub-picosecond intervals for the "
        "reorientational (T1) analysis."
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


def _member_dir(comp: dict, runs_dir: Path, seed) -> Path:
    """Output dir for a member — a per-seed subdir when running replicas, so each
    independent seed keeps its own run and results.json."""
    base = runs_dir / comp["label"]
    return base / f"rep_{seed}" if seed is not None else base


def _run_member(comp: dict, stage: str, runs_dir: Path, seed=None) -> dict:
    from scilink.agents.sim_agents import simulation_pipeline as sp

    model, base_url, api_key = _credentials()
    out_dir = _member_dir(comp, runs_dir, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    # A replica run: stamp this seed into the deck's velocity-create (the MD agent
    # reads SCILINK_MD_SEED), so each seed draws different initial velocities and
    # the runs are independent samples for the Green-Kubo average.
    if seed is not None:
        os.environ["SCILINK_MD_SEED"] = str(seed)

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
        # Human-in-the-loop generation only; nothing executes. Autopilot pauses
        # for approval at each agent decision — override to "autonomous" via
        # UC2_PREVIEW_AUTONOMY for a headless dry run (no input() prompts).
        preview_autonomy = os.environ.get("UC2_PREVIEW_AUTONOMY", "autopilot")
        result = sp.run_complete_workflow(
            _request(comp), autonomy=preview_autonomy,
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


def _analyze_member(comp: dict, runs_dir: Path, seed=None) -> dict:
    from scilink.agents.sim_agents import SimulationAnalysisAgent

    model, base_url, api_key = _credentials()
    out_dir = _member_dir(comp, runs_dir, seed)
    if not out_dir.exists():
        print(f"[{comp['label']}] no run dir; skipping analyze")
        return {}

    agent = SimulationAnalysisAgent(
        api_key=api_key, base_url=base_url, model_name=model,
        output_dir=str(out_dir / "analysis"))
    goal = ("Compute the mass density (g/cm^3), the shear viscosity (mPa*s) by "
            "Green-Kubo, the water self-diffusion coefficient (m^2/s) from the "
            "mean-squared displacement, and the 1H spin-lattice T1 (s) of the water "
            "protons from this finished electrolyte MD run.")
    res = agent.run_analysis(goal, run_dir=str(out_dir))
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"[{comp['label']}] analyze -> {res.get('status')} "
          f"({', '.join(res.get('results', {}) or {}) or 'no properties'})")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["preview", "full", "analyze", "validate", "converge"])
    ap.add_argument("--members", default="all",
                    help="comma-separated labels (e.g. 80-20,70-30) or 'all'")
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    ap.add_argument("--seed", type=int, default=None,
                    help="replica seed: run/analyze this independent copy in a "
                         "rep_<seed> subdir (omit for a single non-replica run)")
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
    if args.stage == "converge":
        # Are enough replicas in for the viscosity TREND to be clear? (TrendCritic)
        import converge_uc2
        converge_uc2.report(comps, runs_dir)
        return

    for comp in comps:
        if args.stage in ("preview", "full"):
            _run_member(comp, args.stage, runs_dir, seed=args.seed)
        elif args.stage == "analyze":
            _analyze_member(comp, runs_dir, seed=args.seed)


if __name__ == "__main__":
    main()
