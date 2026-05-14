"""
Validation harness for the "agents produce correct physics" panel.

Drives each reference crystal (``validation/reference_systems.py``)
through each engine's *real SciLink agent*: the agent generates
relaxation inputs from a natural-language research goal, the run
relaxes the 0.97x-compressed cell, and we compare the relaxed lattice
constant to the experimental value.

The point of the panel is that nothing here writes engine inputs by
hand — VASP inputs come from ``PeriodicDFTAgent``, MLIP relaxations
come from ``MLIPAgent`` delegating to ``MDSimulationAgent``. The
harness only feeds structures + goals in and reads lattice constants
out.

Three phases (``--phase``):
  generate  drive the agents to produce relaxation inputs
  run       execute the relaxations (MLIP in-process; VASP -> sbatch)
  collect   parse relaxed lattice constants, build the comparison table

The panel is run under a ``--mode`` — a benchmark *condition* that dials
how much the agent reasons. Each mode's artifacts live under
``results/<mode>/`` so conditions never collide and ``collect`` can
later compare across them. Modes (added incrementally):
  a_forced         backend forced, prescriptive goals — deterministic
                   MLIP path; the agent-reasoning baseline
  (b_agent_select) agent picks the MLIP backend            [later]
  (c_*_goal …)     goal text dialed prescriptive -> bare   [later]

Typical cluster use:
  python validation/run_panel.py --mode a_forced --phase generate
  python validation/run_panel.py --mode a_forced --phase run    # MLIP only
  sbatch validation/results/a_forced/<system>/vasp/submit.sbatch # per VASP cell
  python validation/run_panel.py --mode a_forced --phase collect
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

# reference_systems.py lives next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_systems import PANEL, ENGINES, get_system   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DEFAULT = os.path.join(_HERE, "results")
_MLIP_ENGINES = ("mace", "chgnet")
_RELAX_SCALE = 0.97          # start every cell ~9% compressed
_FMAX = 0.02                 # eV/A force-convergence threshold
_STEP_CAP = 500              # relaxation step cap

# Benchmark conditions. Each dials how much the agent reasons; adding
# one is a new entry here plus the branch it implies (in generate_mlip
# today; in goal selection once the goal-prescriptiveness axis lands).
_MODES = ("a_forced",)


# ──────────────────────────────────────────────────────────────────
#  paths + structure writers
# ──────────────────────────────────────────────────────────────────

def _engine_dir(out_dir: str, system_name: str, engine: str) -> str:
    d = os.path.join(out_dir, system_name, engine)
    os.makedirs(d, exist_ok=True)
    return d


def _write_poscar(atoms, path: str) -> None:
    from ase.io import write
    write(path, atoms, format="vasp", direct=True)


def _write_lammps_data(atoms, path: str) -> None:
    # The ASE runner reads structures via read_lammps_data(); writing
    # with masses=True lets it recover real element symbols on read.
    from ase.io import write
    write(path, atoms, format="lammps-data", masses=True, atom_style="atomic")


# ──────────────────────────────────────────────────────────────────
#  generate
# ──────────────────────────────────────────────────────────────────

def generate_vasp(system, out_dir, api_key, model_name) -> dict:
    """Drive PeriodicDFTAgent to produce VASP relaxation inputs."""
    from scilink.agents.sim_agents.periodic_dft_agent import PeriodicDFTAgent

    rundir = _engine_dir(out_dir, system.name, "vasp")
    poscar = os.path.join(rundir, "POSCAR")
    _write_poscar(system.build(scale=_RELAX_SCALE), poscar)

    agent = PeriodicDFTAgent(api_key=api_key, model_name=model_name)
    result = agent.generate_inputs(
        structure_file=poscar,
        request=system.research_goal,
        software="vasp",
    )
    if result.get("status") != "success":
        return {"status": "error", "dir": rundir,
                "message": result.get("message", "generate_inputs failed")}

    agent.save_inputs(result, output_dir=rundir)
    # save_inputs may write an echoed POSCAR from input_files; re-write
    # ours so the compressed structure is unambiguously what runs.
    _write_poscar(system.build(scale=_RELAX_SCALE), poscar)
    _write_vasp_sbatch(system, rundir)
    return {
        "status": "generated",
        "dir": rundir,
        "input_files": sorted(result.get("input_files", {})),
        "notes": result.get("notes", ""),
        "run_cmd": "sbatch submit.sbatch",
    }


def generate_mlip(system, engine, out_dir, api_key, model_name,
                  device, mode) -> dict:
    """Drive MLIPAgent -> MDSimulationAgent to produce an MLIP relax run.

    The ``mode`` decides how much the agent reasons:
      a_forced  backend is forced (``backend=engine``) — no LLM
                model-selection; deploy() builds the potential and the
                MD agent generates a deterministic ASE relax script.
    (b_agent_select, when added, drops ``backend=`` so MLIPAgent's
    LLM-driven _select_pretrained_model picks the backend itself.)
    """
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    rundir = _engine_dir(out_dir, system.name, engine)
    structure = os.path.join(rundir, "system.data")
    atoms = system.build(scale=_RELAX_SCALE)
    _write_lammps_data(atoms, structure)

    counts: dict = {}
    for sym in atoms.get_chemical_symbols():
        counts[sym] = counts.get(sym, 0) + 1
    system_info = {"elements": counts, "n_atoms": len(atoms)}

    deploy_kwargs = dict(
        system_info=system_info,
        research_goal=system.research_goal,
        structure_file=structure,
        runner="ase",
        simulation_params={
            "task": "relax",
            "fmax": _FMAX,
            "n_steps": _STEP_CAP,
            "device": device,
        },
    )
    if mode == "a_forced":
        deploy_kwargs["backend"] = engine   # skip LLM model-selection

    agent = MLIPAgent(working_dir=rundir, api_key=api_key,
                      model_name=model_name)
    result = agent.deploy_pretrained(**deploy_kwargs)

    run_script = os.path.basename(result.get("run_path", "run_relax.py"))
    return {
        "status": "generated",
        "dir": rundir,
        "run_script": run_script,
        "backend": result.get("backend"),
        "model_name": result.get("model_name"),
        "run_cmd": f"python {run_script}",
    }


def _write_vasp_sbatch(system, rundir: str) -> None:
    """Per-cell SLURM script: assemble POTCAR, run vasp_std.

    POTCAR is built from $VASP_PP_PATH at submit time rather than
    generated by the agent (pseudopotentials are licensed files that
    live on the cluster, not something the agent should emit).
    """
    elements = " ".join(system.elements)
    script = f"""#!/bin/bash
#SBATCH --job-name=val_{system.name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=02:00:00
#SBATCH --output=vasp.out
#SBATCH --error=vasp.err

# Assemble POTCAR from the cluster pseudopotential library.
: "${{VASP_PP_PATH:?set VASP_PP_PATH to the POTCAR library root}}"
rm -f POTCAR
for el in {elements}; do
    cat "$VASP_PP_PATH/$el/POTCAR" >> POTCAR
done

srun vasp_std
"""
    with open(os.path.join(rundir, "submit.sbatch"), "w") as f:
        f.write(script)


# ──────────────────────────────────────────────────────────────────
#  run  (MLIP only — VASP is submitted via sbatch by the user)
# ──────────────────────────────────────────────────────────────────

def run_mlip(entry: dict, device: str) -> dict:
    """Execute an MLIP relaxation script in-process (subprocess)."""
    rundir = entry["dir"]
    script = entry.get("run_script", "run_relax.py")
    env = dict(os.environ)
    env.setdefault("MACE_DEVICE", device)
    env.setdefault("CHGNET_DEVICE", device)
    proc = subprocess.run(
        [sys.executable, script], cwd=rundir, env=env,
        capture_output=True, text=True,
    )
    ok = proc.returncode == 0
    if not ok:
        (open(os.path.join(rundir, "run_relax.err"), "w")
         .write(proc.stdout + "\n" + proc.stderr))
    return {"ran": ok, "returncode": proc.returncode}


# ──────────────────────────────────────────────────────────────────
#  collect
# ──────────────────────────────────────────────────────────────────

def _lattice_from_vasp(rundir: str):
    """Read the relaxed cubic lattice constant from a VASP CONTCAR."""
    contcar = os.path.join(rundir, "CONTCAR")
    if not os.path.exists(contcar) or os.path.getsize(contcar) == 0:
        return None
    from ase.io import read
    try:
        return float(read(contcar).cell.cellpar()[0])
    except Exception:
        return None


def _lattice_from_mlip(rundir: str):
    """Read the relaxed cubic lattice constant from relax_result.json."""
    rj = os.path.join(rundir, "relax_result.json")
    if not os.path.exists(rj):
        return None
    try:
        with open(rj) as f:
            data = json.load(f)
        return float(data.get("lattice_constant_A"))
    except Exception:
        return None


def collect(out_dir: str, engines, mode: str) -> dict:
    """Parse every run dir, build the comparison table."""
    rows = []
    for system in PANEL:
        row = {
            "system": system.name,
            "crystal": system.crystal,
            "exp_a": system.exp_lattice_constant,
            "engines": {},
        }
        for engine in engines:
            rundir = os.path.join(out_dir, system.name, engine)
            a = (_lattice_from_mlip(rundir) if engine in _MLIP_ENGINES
                 else _lattice_from_vasp(rundir))
            if a is None:
                row["engines"][engine] = {"a": None, "dev_pct": None,
                                          "in_band": None}
            else:
                dev = 100.0 * (a / system.exp_lattice_constant - 1.0)
                row["engines"][engine] = {
                    "a": round(a, 4),
                    "dev_pct": round(dev, 2),
                    # "correct" band: equilibrium at or modestly above
                    # experiment (PBE / PBE-trained MLIPs overestimate).
                    "in_band": -0.5 <= dev <= 2.5,
                }
        rows.append(row)

    summary = {"mode": mode,
               "collected_at": datetime.datetime.now().isoformat(),
               "engines": list(engines), "rows": rows}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_summary_md(out_dir, summary)
    return summary


def _write_summary_md(out_dir: str, summary: dict) -> None:
    engines = summary["engines"]
    head = ["system", "crystal", "exp a (Å)"]
    for e in engines:
        head += [f"{e} a (Å)", f"{e} Δ%"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for row in summary["rows"]:
        cells = [row["system"], row["crystal"], f'{row["exp_a"]:.3f}']
        for e in engines:
            ed = row["engines"][e]
            if ed["a"] is None:
                cells += ["—", "—"]
            else:
                flag = "" if ed["in_band"] else " ⚠"
                cells += [f'{ed["a"]:.4f}', f'{ed["dev_pct"]:+.2f}{flag}']
        lines.append("| " + " | ".join(cells) + " |")
    note = (
        "\n_Δ% is deviation from the experimental lattice constant. "
        "PBE and PBE-trained MLIPs are expected to land slightly above "
        "experiment; ⚠ flags a value outside the [-0.5%, +2.5%] band._\n"
    )
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(f"# Lattice-constant validation panel — mode: "
                f"`{summary['mode']}`\n\n")
        f.write("\n".join(lines) + "\n")
        f.write(note)


# ──────────────────────────────────────────────────────────────────
#  driver
# ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=_MODES, default="a_forced",
        help="Benchmark condition — dials how much the agent reasons. "
             "Artifacts go under results/<mode>/.",
    )
    parser.add_argument(
        "--phase", choices=["generate", "run", "collect", "all"],
        default="all",
    )
    parser.add_argument(
        "--systems", nargs="+", default=[s.name for s in PANEL],
        choices=[s.name for s in PANEL],
    )
    parser.add_argument(
        "--engines", nargs="+", default=list(ENGINES),
        choices=list(ENGINES),
    )
    parser.add_argument("--out-dir", default=_OUT_DEFAULT)
    parser.add_argument(
        "--api-key", default=os.environ.get("SCILINK_API_KEY", ""),
        help="LLM key — required for VASP generation; MLIP generation "
             "forces the backend so a placeholder is fine there.",
    )
    parser.add_argument("--model-name", default="claude-sonnet-4-5")
    parser.add_argument(
        "--device", default="cuda",
        help="Device for MLIP relaxations (cuda / cpu).",
    )
    args = parser.parse_args()

    # Every artifact for this benchmark condition lives under its own
    # results/<mode>/ subtree so conditions never collide.
    out_dir = os.path.join(os.path.abspath(args.out_dir), args.mode)
    os.makedirs(out_dir, exist_ok=True)
    systems = [get_system(n) for n in args.systems]
    do = (["generate", "run", "collect"] if args.phase == "all"
          else [args.phase])

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    # ── generate ──────────────────────────────────────────────────
    if "generate" in do:
        if "vasp" in args.engines and not args.api_key:
            print("ERROR: VASP generation needs --api-key (or "
                  "SCILINK_API_KEY).", file=sys.stderr)
            return 2
        manifest = {"mode": args.mode,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "panel": {}}
        for system in systems:
            print(f"[generate] {system.name}")
            entry = {"exp_lattice_constant": system.exp_lattice_constant,
                     "engines": {}}
            for engine in args.engines:
                print(f"  - {engine}")
                if engine == "vasp":
                    res = generate_vasp(system, out_dir, args.api_key,
                                        args.model_name)
                else:
                    res = generate_mlip(system, engine, out_dir,
                                        args.api_key or "sk-no-llm-needed",
                                        args.model_name, args.device,
                                        args.mode)
                if res.get("status") == "error":
                    print(f"    ERROR: {res.get('message')}")
                entry["engines"][engine] = res
            manifest["panel"][system.name] = entry
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # ── run (MLIP only) ───────────────────────────────────────────
    if "run" in do:
        if not manifest.get("panel"):
            print("ERROR: nothing generated yet — run --phase generate "
                  "first.", file=sys.stderr)
            return 2
        for system in systems:
            entry = manifest["panel"].get(system.name, {})
            for engine in args.engines:
                eng = entry.get("engines", {}).get(engine)
                if not eng or eng.get("status") != "generated":
                    continue
                if engine in _MLIP_ENGINES:
                    print(f"[run] {system.name} / {engine}")
                    r = run_mlip(eng, args.device)
                    eng.update(r)
                    if not r["ran"]:
                        print(f"    FAILED (rc={r['returncode']}) — see "
                              f"run_relax.err")
                else:
                    print(f"[run] {system.name} / vasp — submit "
                          f"{eng['dir']}/submit.sbatch")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # ── collect ───────────────────────────────────────────────────
    if "collect" in do:
        summary = collect(out_dir, args.engines, args.mode)
        print()
        with open(os.path.join(out_dir, "summary.md")) as f:
            print(f.read())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
