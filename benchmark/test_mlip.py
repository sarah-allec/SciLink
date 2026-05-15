"""MLIP→MD delegation test — for each system tagged ``mlip``, exercise
the ``MLIPAgent.deploy_pretrained → DeployedPotential → MDSimulationAgent``
handoff and score:

  * Deploy succeeded (no exception, ``status == "success"``).
  * DeployedPotential descriptor is well-formed
    (``backend`` + ``model_name`` + ``elements`` + ``ase_calculator``).
  * MD agent produced a runnable script at ``run_path``.
  * **Bonus signal:** which backend did the agent pick?  The router
    test already showed the agent picks CHGNet for battery contexts
    and MACE for generic crystals — this is where we observe the
    deploy-time choice on a wider system roster.

Three phases mirroring test_dft.py:

  generate   — build structure, write LAMMPS data, call deploy_pretrained,
               record backend choice + descriptor + run_path.
  submit     — sbatch the generated runner (GPU partition).
  collect    — parse trajectory; score convergence + sanity.
  all        — generate + submit (collect runs after the jobs finish).

Examples
--------
    python -m benchmark.runner test_mlip --phase generate --system fe_bcc
    python -m benchmark.runner test_mlip --phase generate           # all
    python -m benchmark.runner test_mlip --phase submit
    python -m benchmark.runner test_mlip --phase collect
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._score import CaseResult, RunnerManifest, write_manifest, write_summary_md
from .systems import BenchmarkSystem, get_system, systems_for_agent


_OUT_DEFAULT = "benchmark/outputs/test_mlip"


# ──────────────────────────────────────────────────────────────────
#  Generate phase
# ──────────────────────────────────────────────────────────────────

def _build_atoms_for_mlip(system: BenchmarkSystem):
    """Return ASE Atoms for systems whose build() yields Atoms; raise
    NotImplementedError for liquid / Packmol-spec systems (handled
    later)."""
    obj = system.build()
    # Packmol spec sentinel
    if isinstance(obj, dict) and obj.get("kind") == "packmol":
        raise NotImplementedError(
            f"{system.name!r} is a Packmol spec; "
            f"test_mlip does not yet materialise liquid boxes"
        )
    return obj


def _write_lammps_data(atoms, path: str) -> None:
    """ASE's lammps-data writer with masses=True so the ASE reader on the
    other side recovers real element symbols from the atomic masses
    (the trick the MLIP runner scripts already use)."""
    from ase.io import write
    write(path, atoms, format="lammps-data", masses=True, atom_style="atomic")


def _generate_one(system: BenchmarkSystem,
                  rundir: str,
                  api_key: Optional[str],
                  base_url: Optional[str],
                  model_name: str,
                  task: str,
                  temperature: float,
                  n_steps: int,
                  runner: str) -> Dict[str, Any]:
    """Deploy a pretrained MLIP for ``system`` (agent picks backend),
    generating the LAMMPS-data file and the MD runner script in
    ``rundir``."""
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    os.makedirs(rundir, exist_ok=True)

    atoms = _build_atoms_for_mlip(system)
    data_file = os.path.join(rundir, "system.data")
    _write_lammps_data(atoms, data_file)

    elements = sorted(set(atoms.get_chemical_symbols()))
    composition = dict(Counter(atoms.get_chemical_symbols()))
    system_info = {
        "elements":       elements,
        "n_atoms":        len(atoms),
        "composition":    composition,
        "structure_type": system.kind,
        "description":    system.description,
    }
    # Goal is the system's bare-level prompt plus an MLIP-aware
    # addendum so the agent has the right context for backend choice.
    base_goal = system.goal(level="bare")
    research_goal = (
        f"{base_goal}  "
        f"Use a pretrained machine-learning interatomic potential "
        f"(your choice — pick the model whose training distribution "
        f"best fits this system).  Run a {task.upper()} simulation at "
        f"{temperature:g} K."
    )

    agent = MLIPAgent(api_key=api_key, base_url=base_url,
                      model_name=model_name)
    deploy_result = agent.deploy_pretrained(
        system_info=system_info,
        research_goal=research_goal,
        simulation_params={
            "task":        task,
            "temperature": temperature,
            "n_steps":     n_steps,
            "device":      "cuda",
        },
        runner=runner,
        structure_file=data_file,
        # backend=None → let the agent pick
    )

    return {
        "status":          deploy_result.get("status", "success"),
        "backend":         deploy_result.get("backend"),
        "model_name":      deploy_result.get("model_name"),
        "model_file":      deploy_result.get("model_file"),
        "elements":        deploy_result.get("elements"),
        "selection_notes": (deploy_result.get("selection") or {}).get("notes", ""),
        "runner":          deploy_result.get("runner", runner),
        "task":            deploy_result.get("task", task),
        "run_path":        deploy_result.get("run_path"),
        "rundir":          rundir,
        "data_file":       data_file,
        "notes":           deploy_result.get("notes", ""),
    }


# ──────────────────────────────────────────────────────────────────
#  Submit phase  —  sbatch the runner script
# ──────────────────────────────────────────────────────────────────

def _make_per_cell_sbatch(rundir: str, run_path: str, job_name: str) -> str:
    """Write a thin sbatch wrapper around the deploy-emitted runner.

    The script the MD agent produces is either a Python file
    (ASE runner) or a LAMMPS input file (LAMMPS runner).  We detect
    via extension and dispatch accordingly.
    """
    script_path = os.path.join(rundir, "submit.sbatch")
    if run_path.endswith(".py"):
        run_cmd = f'"$CONDA_PREFIX/bin/python" "$RUN_FILE"'
    elif run_path.endswith((".lmp", ".in")):
        run_cmd = 'lmp -in "$RUN_FILE"'
    else:
        run_cmd = '"$RUN_FILE"'

    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=a100_shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=mlip.out
#SBATCH --error=mlip.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source /people/alle927/scilink_mace_env.sh

export MACE_DEVICE=cuda
RUN_FILE="{os.path.basename(run_path)}"

{run_cmd}
"""
    with open(script_path, "w") as f:
        f.write(content)
    return script_path


def _sbatch(rundir: str) -> tuple[bool, str]:
    script = os.path.join(rundir, "submit.sbatch")
    if not os.path.isfile(script):
        return False, f"no submit.sbatch in {rundir}"
    try:
        out = subprocess.check_output(
            ["sbatch", "--parsable", "submit.sbatch"],
            cwd=rundir, stderr=subprocess.STDOUT, text=True)
        return True, out.strip()
    except FileNotFoundError:
        return False, "sbatch not in PATH (login node only?)"
    except subprocess.CalledProcessError as exc:
        return False, f"sbatch failed: {exc.output.strip()}"


# ──────────────────────────────────────────────────────────────────
#  Collect phase  —  score the descriptor + run state
# ──────────────────────────────────────────────────────────────────

def _has_traj_artifact(rundir: str) -> bool:
    """Best-effort detection of an MD/relax trajectory output.  Doesn't
    parse — just confirms the runner wrote *something*."""
    p = Path(rundir)
    for name in ("relax.log", "md.log", "trajectory.xyz", "trajectory.dump",
                 "log.lammps", "lammps.log", "OUTCAR"):
        if (p / name).exists() and (p / name).stat().st_size > 0:
            return True
    return False


def _score_one(system: BenchmarkSystem,
               cell: Dict[str, Any]) -> CaseResult:
    """Score the deploy + run-generation for one system.  Up to 1.0:
      * 0.30  deploy returned status=='success'
      * 0.30  descriptor well-formed (backend + model_name + elements)
      * 0.30  run_path was produced (and exists on disk)
      * 0.10  trajectory artifact exists (only checked if cell submitted)
    """
    detail: Dict[str, Any] = {"backend": cell.get("backend"),
                              "model_name": cell.get("model_name")}
    score = 0.0

    if cell.get("status") == "success":
        score += 0.30
        detail["deploy_status"] = "success"
    else:
        detail["deploy_status"] = cell.get("status", "missing")
        detail["error"] = cell.get("message") or cell.get("error")

    if (cell.get("backend") and cell.get("model_name")
            and cell.get("elements")):
        score += 0.30
        detail["descriptor_ok"] = True
    else:
        detail["descriptor_ok"] = False

    run_path = cell.get("run_path")
    if run_path and os.path.exists(run_path):
        score += 0.30
        detail["run_script_exists"] = True
    else:
        detail["run_script_exists"] = False

    if _has_traj_artifact(cell.get("rundir", "")):
        score += 0.10
        detail["trajectory_artifact"] = True
    else:
        detail["trajectory_artifact"] = False

    passed = score >= 0.70    # deploy + descriptor + run_path all true
    return CaseResult(
        id=system.name,
        expected={"agents": ["mlip", "md"], "elements": system.elements},
        actual={"rundir": cell.get("rundir"), **detail},
        score=round(score, 2),
        passed=passed,
        notes=cell.get("selection_notes", ""),
    )


# ──────────────────────────────────────────────────────────────────
#  Manifest persistence (3-phase state file)
# ──────────────────────────────────────────────────────────────────

def _manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, "manifest.json")


def _load_state(out_dir: str) -> Dict[str, Any]:
    p = _manifest_path(out_dir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"runner": "test_mlip", "mode": "cluster",
            "n_cases": 0, "passed": 0, "failed": 0,
            "metrics": {}, "cases": [], "cells": {}}


def _save_state(out_dir: str, state: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(_manifest_path(out_dir), "w") as f:
        json.dump(state, f, indent=2)


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def _selected_systems(args) -> List[BenchmarkSystem]:
    systems = systems_for_agent("mlip")
    if args.system:
        systems = [get_system(args.system)]
    if args.skip_liquids:
        systems = [s for s in systems if s.kind != "liquid"]
    if args.limit:
        systems = systems[: args.limit]
    return systems


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="test_mlip", description=__doc__)
    parser.add_argument("--phase", choices=("generate", "submit", "collect", "all"),
                        default="all")
    parser.add_argument("--model",    default="claude-sonnet-4-5")
    parser.add_argument("--api-key",  default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--system",   default=None)
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--task",     choices=("md", "relax"), default="md")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--n-steps",  type=int, default=1000)
    parser.add_argument("--runner",   default="lammps",
                        choices=("lammps", "ase"))
    parser.add_argument("--skip-liquids", action="store_true",
                        help="skip Packmol-spec systems (test_mlip doesn't "
                             "materialise liquid boxes yet; default keeps "
                             "them so the skip is logged loudly)")
    parser.add_argument("--out-dir", default=_OUT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    systems = _selected_systems(args)
    print(f"test_mlip :: phase={args.phase}  n_systems={len(systems)}  "
          f"task={args.task}  runner={args.runner}  model={args.model}")

    if args.dry_run:
        for s in systems:
            print(f"  [{s.kind:<8}] {s.name:<22} elements={s.elements}  "
                  f"description={s.description[:60]}")
        return 0

    state = _load_state(args.out_dir)
    do = {"all": {"generate", "submit", "collect"}}.get(
        args.phase, {args.phase})

    api_key = args.api_key or os.environ.get("SCILINK_API_KEY")
    base_url = args.base_url or os.environ.get("SCILINK_BASE_URL")

    # ── generate ─────────────────────────────────────────────────
    if "generate" in do:
        if not api_key:
            print("!! no SCILINK_API_KEY — generate needs LLM access",
                  file=sys.stderr)
            return 2
        for s in systems:
            rundir = os.path.join(args.out_dir, s.name)
            print(f"[generate] {s.name}")
            try:
                cell = _generate_one(
                    s, rundir, api_key, base_url, args.model,
                    args.task, args.temperature, args.n_steps, args.runner,
                )
                print(f"    backend chosen: {cell.get('backend')}  "
                      f"model: {cell.get('model_name')}  "
                      f"run_path: {cell.get('run_path')}")
            except NotImplementedError as exc:
                print(f"    skipped: {exc}")
                cell = {"status": "skipped", "rundir": rundir,
                        "message": str(exc)}
            except Exception as exc:
                print(f"    !! {exc!r}")
                traceback.print_exc()
                cell = {"status": "error", "rundir": rundir,
                        "message": repr(exc)}
            state["cells"][s.name] = cell
        _save_state(args.out_dir, state)

    # ── submit ───────────────────────────────────────────────────
    if "submit" in do:
        for s in systems:
            cell = state["cells"].get(s.name) or {}
            if cell.get("status") != "success" or not cell.get("run_path"):
                print(f"[submit] {s.name}: skipping (status="
                      f"{cell.get('status', 'missing')})")
                continue
            _make_per_cell_sbatch(
                cell["rundir"], cell["run_path"],
                job_name=f"benchmlip_{s.name}",
            )
            ok, msg = _sbatch(cell["rundir"])
            if ok:
                cell["job_id"] = msg
                cell["status"] = "submitted"
                print(f"[submit] {s.name} → job {msg}")
            else:
                cell["submit_error"] = msg
                print(f"[submit] {s.name}: FAILED — {msg}")
            state["cells"][s.name] = cell
        _save_state(args.out_dir, state)

    # ── collect ──────────────────────────────────────────────────
    if "collect" in do:
        manifest = RunnerManifest(runner="test_mlip", mode="cluster")
        for s in systems:
            cell = state["cells"].get(s.name)
            if not cell or cell.get("status") in ("skipped", "error", "missing"):
                manifest.append(CaseResult(
                    id=s.name,
                    expected={"agents": ["mlip", "md"]},
                    actual=cell or {"status": "missing"},
                    score=0.0, passed=False,
                    notes=cell.get("message", "no cell record") if cell else "",
                ))
                continue
            case = _score_one(s, cell)
            manifest.append(case)
            flag = "✓" if case.passed else ("~" if case.score > 0 else "✗")
            print(f"[collect] {flag} {s.name:<22} backend={case.actual.get('backend')}"
                  f"  score={case.score}")

        if manifest.cases:
            # Track which backends got picked — the headline signal
            picks: Counter = Counter()
            for c in manifest.cases:
                b = c.actual.get("backend")
                if b:
                    picks[b] += 1
            manifest.metrics = {
                "pass_rate":  manifest.passed / manifest.n_cases,
                "mean_score": sum(c.score for c in manifest.cases) / manifest.n_cases,
                "backend_picks": dict(picks),
            }
        write_manifest(args.out_dir, manifest)
        write_summary_md(args.out_dir, manifest)
        print(f"\nwrote {args.out_dir}/manifest.json (+summary.md)")
        if manifest.metrics:
            print(f"pass rate:    {manifest.metrics['pass_rate']:.2f}")
            print(f"mean score:   {manifest.metrics['mean_score']:.2f}")
            print(f"backend picks: {manifest.metrics['backend_picks']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
